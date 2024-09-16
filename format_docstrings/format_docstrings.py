from __future__ import annotations

import ast

from griffe.dataclasses import Function
from griffe.extensions import VisitorExtension

from novae.utils._docs import format_docstring


class DocParamsExtension(VisitorExtension):
    def visit_functiondef(self, node: ast.FunctionDef) -> None:
        function: Function = self.visitor.current.members[node.name]  # type: ignore[assignment]
        if hasattr(function, "decorators"):
            for decorator in function.decorators:
                if str(decorator.value).endswith("format_docs"):
                    function.docstring.value = format_docstring(
                        function.docstring.value,
                    )


Extension = DocParamsExtension
