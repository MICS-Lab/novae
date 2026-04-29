import json
import logging
import warnings
from os import getenv

import pandas as pd
from anndata import AnnData

from .. import plot, utils
from .._constants import Keys

log = logging.getLogger(__name__)


def _create_prompt(tissue: str = "unknown", species: str | None = None, spatial_context: str | None = None) -> str:
    """
    Prompt for domain annotation.
    """

    species_text = species if species else ""
    spatial_context_text = spatial_context if spatial_context else ""

    return (
        f"You are an expert in spatial transcriptomics analysis specializing in {species_text} tissue domain annotation. "
        f"Identify the most likely spatial domain name or tissue region (niche) for each domain of a {tissue} tissue based on marker genes and potentially enriched pathway scores. "
        "Consider spatial context, functional zones, and tissue organization when assigning domain names. "
        f"{spatial_context_text} "
        "Be concise but specific. Some domain may represent mixed or transitional regions. "
        "CRITICAL OUTPUT RULES: "
        "- The 'domain_name' must contain ONLY a short domain label. "
        "- Do NOT label domain using cell-type names, but using niche names. For instance, don’t label a 'B and T cells' domain, but use 'Tertiary Lymphoid Structure' instead. "
        "- Do NOT include explanations, examples, or additional details. "
        "- Do NOT use phrases like 'including', 'such as', or 'with'. "
        "- Do NOT skip any domain. "
        "- Do NOT add explanations."
        "- Do NOT add explanations. "
        "Return only valid JSON matching the provided schema."
    )


def _format_pathway_scores(
    pathway_scores: pd.DataFrame | None,
    domain_ids: list[int] | list[str],
) -> str:
    if pathway_scores is None:
        return ""

    lines = []
    for domain_id in domain_ids:
        values = ", ".join(f"{name}={value:.4f}" for name, value in pathway_scores.loc[domain_id].items())
        lines.append(f"Domain {domain_id}: {values}")

    return "Pathway scores:\n" + "\n".join(lines)


def _format_domain_cell_percentages(adata: AnnData, obs_key: str, domain_ids: list[int] | list[str]) -> str:
    obs_as_str = adata.obs[obs_key].astype(str)
    proportions = obs_as_str.value_counts(normalize=True)

    lines = []
    for domain_id in domain_ids:
        domain_id_str = str(domain_id)
        pct = proportions.get(domain_id_str, 0.0)
        lines.append(f"Domain {domain_id}: {pct:.2%}")

    return "Cell percentages by domain:\n" + "\n".join(lines)


def _output_schema(
    domain_ids: list,
    domain_key: str,
    label_key: str,
    confidence_score_key:str,
    additionalProperties: bool = False,
) -> dict:
    schema = {
        "type": "object",
        "properties": {
            label_key: {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        domain_key: {"type": "string", "enum": domain_ids},
                        label_key: {
                            "type": "string",
                            "description": "Most likely domain name. May be a mixed label if needed.",
                        },
                        confidence_score_key: {
                            "type": "number",
                            "description": "A confidence score between 0 and 1 for the annotation.",
                        },
                    },
                    "required": [domain_key, label_key, confidence_score_key],
                    "additionalProperties": additionalProperties,
                },
            }
        },
        "required": [label_key],
        "additionalProperties": additionalProperties,
    }
    return schema


def _validate_api_key(
    api_key: str | None,
    env_var: str | None = None,
    provider: str | None = None,
) -> str:
    if api_key is None:
        warnings.warn(
            f"`api_key` was not provided. Trying environment variable `{env_var}`.",
            stacklevel=2,
        )
        api_key = getenv(env_var)
        if api_key is None or not isinstance(api_key, str) or not api_key.strip():
            raise ValueError(f"{provider} API key is required. Provide `api_key` or set `{env_var}`.")
        return api_key.strip()
    else:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("`api_key` must be a non-empty string when provided.")
        return api_key.strip()


def _get_api_request_func(provider: str, model: str) -> callable:
    model_name = model.strip() if isinstance(model, str) else ""
    if not model_name:
        raise ValueError("`model` must be a non-empty string (e.g. `gpt-4.1` or `claude-sonnet-4-5`).")

    provider_name = provider.strip().lower() if isinstance(provider, str) else ""
    if provider_name not in {"openai", "anthropic"}:
        raise ValueError("`provider` must be one of: 'openai', 'anthropic'.")

    api_request_func = _Anthropic_api_request if provider_name == "anthropic" else _OpenAI_api_request
    return api_request_func


def _OpenAI_api_request(
    model: str,
    api_key: str | None,
    messages: list[dict[str, str]],
    output_schema: dict,
    max_tokens: int,
    seed: int | None = None,
) -> json:
    try:
        from openai import OpenAI
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing optional dependency `openai` required for `novae.utils.annotate_domains`. "
            "Please install it with `pip install openai`."
        ) from e

    client = OpenAI(api_key=api_key)

    response_format = {
        "type": "json_schema",
        "json_schema": {"name": Keys.LABEL_SUFFIX, "schema": output_schema, "strict": True},
    }

    request_kwargs = {
        "model": model,
        "messages": messages,
        "response_format": response_format,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        request_kwargs["seed"] = seed

    try:
        response = client.chat.completions.create(**request_kwargs)
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        raise RuntimeError(f"OpenAI API request failed: {e}") from e


def _Anthropic_api_request(
    model: str,
    api_key: str | None,
    messages: list[dict[str, str]],
    max_tokens: int,
    output_schema: dict,
    seed: int | None = None,
) -> dict:
    try:
        import anthropic
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing optional dependency `anthropic` required for `novae.utils.annotate_domains`. "
            "Please install with `pip install anthropic`."
        ) from e

    client = anthropic.Anthropic(api_key=api_key)

    if max_tokens is None:
        max_tokens = 1000
    elif not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("`max_tokens` must be a positive integer.")

    output_config = {
        "format": {
            "type": "json_schema",
            "schema": output_schema,
        }
    }

    system = "\n\n".join(message["content"] for message in messages if message["role"] == "developer")
    user_messages = [
        {"role": "user", "content": message["content"]} for message in messages if message["role"] == "user"
    ]

    request_kwargs = {
        "model": model,
        "messages": user_messages,
        "max_tokens": max_tokens,
        "system": system,
        "output_config": output_config,
    }
    if seed is not None:
        request_kwargs["seed"] = seed

    try:
        response = client.messages.create(**request_kwargs)
        return json.loads(response.content[0].text)
    except Exception as e:
        raise RuntimeError(f"Anthropic API request failed: {e}") from e


def label_domains(
    adata: AnnData | None = None,
    pathways: dict[str, list[str]] | str | None = None,
    obs_key: str | None = None,
    provider: str = "openai",
    model: str = "gpt-4.1",
    api_key: str | None = None,
    tissue: str = "unknown",
    species: str | None = None,
    n_genes: int = 15,
    spatial_context: str | None = None,
    return_prompt: bool = False,
    max_tokens: int = 1024,
    key_added: str | None = None,
    seed: int | None = None,
) -> pd.DataFrame | dict[str, object]:
    """Annotate spatial domains with an LLM using domain marker genes.

    !!! info
        One annotation is generated per domain and stored in `adata.obs[key_added]`
        (or `adata.obs["novae_domains_X_annotation"]` when `key_added` is not provided).

    !!! note
        `obs_key` must reference an existing domain column in `adata.obs`,
        typically created with [novae.Novae.assign_domains].

    Args:
        adata: An `AnnData` object, or a list of `AnnData` objects. Optional if the model was initialized with `adata`.
        pathways: Either a dictionary of pathways (keys are pathway names, values are lists of gene names), or a path to a [GSEA](https://www.gsea-msigdb.org/gsea/msigdb/index.jsp) JSON file.
        obs_key: Key in `adata.obs` containing domain IDs to annotate. By default, it will use the last available Novae domain key.
        provider: LLM provider to use. Supported providers: 'openai', 'anthropic'.
        model: OpenAI model name used for annotation.
        api_key: OpenAI API key. If `None`, uses `OPENAI_API_KEY` from the environment.
        tissue: Tissue name (for example, `"liver"`).
        species: Species name (for example, `"human"` or `"mouse"`).
        n_genes: Number of marker genes per domain passed to the LLM prompt.
        spatial_context: Optional biological/spatial context to include in the prompt.
        return_prompt: If `True`, returns only the generated request payload (`messages` and `output_schema`) so you can copy/paste it into an LLM manually; no LLM request is made.
        key_added: Output key used to store annotations in `adata.obs`.
        seed: Optional random seed passed to the annotation utility.
        max_tokens: Maximum number of tokens the model is allowed to generate for the annotation response.,

    Returns:
        A DataFrame with domain annotations. If `return_prompt=True`, returns a dictionary containing `messages` and `output_schema`.
    """

    obs_key = utils.check_available_domains_key([adata], obs_key)

    key_added = f"{obs_key}_{Keys.LABEL_SUFFIX}" if key_added is None else key_added

    gene_marker_dict = utils.markers_as_dict(adata, n_genes)

    domain_ids = list(gene_marker_dict.keys())

    input_markers = "Gene markers:\n" + "\n".join(
        f"Domain {domain_id}: {', '.join(gene_marker_dict[domain_id])}" for domain_id in domain_ids
    )
    input_percentages = _format_domain_cell_percentages(adata, obs_key, domain_ids)

    pathway_scores = (
        None
        if pathways is None
        else plot.pathway_scores(adata, obs_key=obs_key, pathways=pathways, show=False, return_df=True)
    )

    input_pathway = _format_pathway_scores(pathway_scores, domain_ids)

    prompt = _create_prompt(species=species, tissue=tissue, spatial_context=spatial_context)

    messages = [
        {
            "role": "developer",
            "content": prompt,
        },
        {
            "role": "user",
            "content": (
                f"Annotate the following domains.\n\n{input_markers}\n\n{input_percentages}\n\n{input_pathway}"
            ),
        },
    ]

    output_schema = _output_schema(
                    domain_ids=domain_ids,
                    domain_key=obs_key,
                    label_key=Keys.LABEL_SUFFIX,
                    confidence_score_key=Keys.CONFIDENCE_SCORE)

    if return_prompt:
        return {"messages": messages, "output_schema": output_schema}

    is_openai = provider.lower().startswith("openai")

    api_key = _validate_api_key(
        api_key,
        env_var=Keys.OPENAI_API_KEY if is_openai else Keys.ANTHROPIC_API_KEY,
        provider=provider,
    )

    api_request_func = _get_api_request_func(provider=provider, model=model)

    result = api_request_func(
        model=model,
        api_key=api_key,
        messages=messages,
        max_tokens=max_tokens,
        output_schema=output_schema,
        seed=seed,
    )

    domain_ann = {d[obs_key]: d[Keys.LABEL_SUFFIX] for d in result[Keys.LABEL_SUFFIX]}

    adata.obs[key_added] = adata.obs[Keys.LABEL_SUFFIX].map(domain_ann)
    log.info(f"Added: {key_added}")

    return domain_ann #pd.DataFrame(result[Keys.LABEL_SUFFIX])
