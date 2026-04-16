import json
import warnings
from os import getenv

import pandas as pd
from openai import OpenAI

from .._constants import Keys


def _create_prompt(tissue: str = "unknown", species: str | None = None, spatial_context: str | None = None) -> str:
    """
    Prompt for domain annotation.
    """

    return (
        "You are an expert in spatial transcriptomics analysis specializing in {species} tissue domain annotation. "
        "Identify the most likely spatial domain name or tissue region (niche) for each domain of a {tissue} tissue based on marker genes and potentially enriched pathway scores. "
        "Consider spatial context, functional zones, and tissue organization when assigning domain names. "
        "{spatial_context} "
        "Be concise but specific. Some domain may represent mixed or transitional regions. "
        "CRITICAL OUTPUT RULES: "
        "- The 'domain_name' must contain ONLY a short domain label. "
        "- Do NOT include parentheses. "
        "- Do NOT include explanations, examples, or additional details. "
        "- Do NOT use phrases like 'including', 'such as', or 'with'. "
        "- Do NOT skip any domain. "
        "- Do NOT add explanations."
        "Return only valid JSON matching the provided schema."
    ).format(
        species=species if species else "", tissue=tissue, spatial_context=spatial_context if spatial_context else ""
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


def _output_schema(
    domain_ids: list,
    additionalProperties: bool = False,
) -> str:
    schema = {
        "type": "object",
        "properties": {
            Keys.DOMAIN_ANNOTATION: {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        Keys.DOMAIN_ID: {"type": "string", "enum": domain_ids},
                        Keys.DOMAIN_ANNOTATION: {
                            "type": "string",
                            "description": "Most likely domain name. May be a mixed label if needed.",
                        },
                        Keys.CONFIDENCE_SCORE: {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "A confidence score between 0 and 1 for the annotation.",
                        },
                    },
                    "required": [Keys.DOMAIN_ID, Keys.DOMAIN_ANNOTATION, Keys.CONFIDENCE_SCORE],
                    "additionalProperties": additionalProperties,
                },
            }
        },
        "required": [Keys.DOMAIN_ANNOTATION],
        "additionalProperties": additionalProperties,
    }
    return schema


def _validate_api_key(api_key: str | None) -> str:
    if api_key is None:
        warnings.warn(
            "`api_key` was not provided. Trying environment variable `{Keys.OPENAI_API_KEY}`.",
            stacklevel=2,
        )
        api_key = getenv(Keys.OPENAI_API_KEY)
        if api_key is None or not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("OpenAI API key is required. Provide `api_key` or set `OPENAI_API_KEY`.")
        return api_key.strip()
    else:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("`api_key` must be a non-empty string when provided.")
        return api_key.strip()


def _OpenAI_api_request(
    model: str,
    api_key: str | None,
    messages: list[dict[str, str]],
    response_format: dict,
    seed: int | None = None,
) -> json:
    client = OpenAI(api_key=api_key)

    request_kwargs = {
        "model": model,
        "messages": messages,
        "response_format": response_format,
    }
    if seed is not None:
        request_kwargs["seed"] = seed

    try:
        response = client.chat.completions.create(**request_kwargs)
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        raise RuntimeError(f"OpenAI API request failed: {e}") from e


def annotate_domains(
    marker_dict: dict[str : list[str]],
    pathway_scores: pd.DataFrame | None = None,
    tissue: str = "unknown",
    species: str | None = None,
    spatial_context: str | None = None,
    model: str = "gpt-4.1",
    api_key: str | None = None,
    seed: int | None = None,
) -> json:
    """
    Ask the model for one annotation per domain and return parsed JSON.
    Args:
        marker_dict: Dictionary mapping domain_ids to lists of marker genes
        tissue: Tissue name (e.g., 'liver')
        species: Species name (e.g., 'human', 'mouse')
        spatial_context: context to include in the prompt
        model: name of OpenAI model to use
        api_key: OpenAI API key. If `None`, the function tries `OPENAI_API_KEY` from the environment and raises an error if it is missing.
        seed: Optional random seed passed to the annotation utility.

    Returns:
        json: domain annotations generated by the model in json format
    """
    domain_ids = list(marker_dict.keys())

    input_markers = "Gene markers:\n" + "\n".join(
        f"Domain {domain_id}: {', '.join(marker_dict[domain_id])}" for domain_id in domain_ids
    )

    input_pathway = _format_pathway_scores(pathway_scores, domain_ids)

    api_key = _validate_api_key(api_key)

    response_format = {
        "type": "json_schema",
        "json_schema": {"name": Keys.DOMAIN_ANNOTATION, "schema": _output_schema(domain_ids), "strict": True},
    }

    messages = [
        {"role": "developer", "content": _create_prompt(species, tissue, spatial_context)},
        {"role": "user", "content": (f"Annotate the following domains.\n\n{input_markers}\n\n{input_pathway}")},
    ]

    return _OpenAI_api_request(
        model=model,
        api_key=api_key,
        messages=messages,
        response_format=response_format,
        seed=seed,
    )
