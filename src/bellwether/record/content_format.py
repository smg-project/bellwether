"""The content format a chat template takes a message's content in, as vLLM selects it, and the messages in it.

vLLM hands a template a message's content in one of two formats, chosen per template by reading its source
(``_detect_content_format`` in ``vllm/renderers/hf.py:489`` at v0.31.0, the release whose image bellwether pins; the
``--chat-template-content-format`` option names the two): a template that loops over a message's content takes the
"openai" format, a list of typed parts, and string content reaches it as a one-item text part list; every other template
takes the "string" format, the text itself. The reference renders in the format vLLM selects, so a template whose two
branches differ (a separator after every part, a truthiness check, a concatenation only a string survives) is rendered
as the engine renders it for the same request, and a template that renders both the same is rendered as before.

The detection is the engine's, function for function (``hf.py:321-502``). A message is a loop variable over
``messages`` or over a variable assigned from it. The template takes the "openai" format when it loops over
``message.content`` (also through a filter, a test or a slice), over a macro parameter it fills with a message's content
by position or by keyword, or, outside any macro, over a variable named ``content``. A type test, a ``length`` filter or
an index on the content is not a loop over it. A template that does not parse, or whose walk meets a loop or assignment
target that is not a plain name where the engine asserts one, is of the "string" format, the engine's default. The
engine resolves the template per request, so a checkpoint with named templates is detected on the one transformers
applies for the request.

The messages are converted as ``_parse_chat_message_content`` in ``vllm/entrypoints/chat_utils.py:2007`` converts them
for the "openai" format: a string becomes ``[{"type": "text", "text": ...}]``; a missing or null content an empty list;
in a list, a bare string becomes a text part and a text part keeps its ``text`` and its other fields
(``_parse_chat_message_content_part``, ``:1875``). A tool message is left as the request gives it: the engine joins a
tool result's text parts back into one string (``:2049-2064``), so its string never reaches the template as parts. A
part of another type is a media part, which the engine resolves and the reference cannot, so the case is rejected. In
the "string" format the content is left as the request gives it: a string is what the engine hands through (the text
parts of a message joined with newlines, ``_parse_chat_message_content_parts``, ``:1814``, which leaves a string as it
is). Nothing else of the engine's pipeline is applied here: tool call arguments, the developer role and the thinking
fields reach the template as the request gives them.
"""

from __future__ import annotations

import copy
from collections import deque
from collections.abc import Iterator
from functools import lru_cache
from typing import Literal

import jinja2
import jinja2.nodes

ContentFormat = Literal["string", "openai"]
TEXT_PART_TYPES = ("text", "input_text", "output_text")


@lru_cache(maxsize=64)
def detect_content_format(template: str) -> ContentFormat:
    """The content format vLLM selects for ``template``: "openai" when it loops over a message's content, else "string",
    which is also the engine's default for a template it cannot parse or walk."""
    ast = _parse(template)
    if ast is None:
        return "string"
    try:
        next(_loops_over_content(ast))
    except StopIteration:
        return "string"
    except Exception:  # the engine's walk asserts plain names; where it fails, the engine falls back to its default
        return "string"
    return "openai"


def in_content_format(messages: list[dict], content_format: ContentFormat) -> list[dict]:
    """A copy of ``messages`` with each message's content in ``content_format``, as the engine hands them to the
    template.

    Only the "openai" format converts anything; the "string" format is the request as given. A message is copied only
    when its content changes, so the request's own objects are never written to.
    """
    if content_format != "openai":
        return messages
    return [_in_openai_format(message) for message in messages]


def _in_openai_format(message: dict) -> dict:
    if message.get("role") == "tool":
        return message
    content = message.get("content")
    if isinstance(content, str):
        parts = [{"type": "text", "text": content}]
    elif content is None:
        parts = []
    elif isinstance(content, list):
        parts = [_text_part(part) for part in content]
    else:
        raise ValueError(f"a message's content is neither a string nor a list of parts: {type(content).__name__}")
    return {**copy.deepcopy(message), "content": parts}


def _text_part(part) -> dict:
    """A text part as the engine wraps it; any other part is a media part the reference cannot resolve."""
    if isinstance(part, str):
        return {"type": "text", "text": part}
    if isinstance(part, dict) and part.get("type") in TEXT_PART_TYPES and isinstance(part.get("text"), str):
        return {"type": "text", "text": part["text"], **{k: v for k, v in part.items() if k not in ("type", "text")}}
    kind = part.get("type") if isinstance(part, dict) else type(part).__name__
    raise ValueError(f"a content part of type {kind!r} is not text; the reference cannot render media parts")


def _parse(template: str) -> jinja2.nodes.Template | None:
    """The template's syntax tree from the environment transformers renders with, as the engine reads it; None when the
    template does not compile."""
    import transformers.utils.chat_template_utils as hf_chat_utils

    try:
        compiled = hf_chat_utils._compile_jinja_template(template)
        return compiled.environment.parse(template)
    except Exception:
        return None


# The walk below is vLLM's (vllm/renderers/hf.py:321-475 at v0.31.0), with its names kept so the two can be read side by
# side; its assertions are the engine's and are what makes a template with an unusual target fall back to "string".


def _is_var_access(node: jinja2.nodes.Node, varname: str) -> bool:
    if isinstance(node, jinja2.nodes.Name):
        return node.ctx == "load" and node.name == varname
    return False


def _is_attr_access(node: jinja2.nodes.Node, varname: str, key: str) -> bool:
    if isinstance(node, jinja2.nodes.Getitem):
        return _is_var_access(node.node, varname) and isinstance(node.arg, jinja2.nodes.Const) and node.arg.value == key
    if isinstance(node, jinja2.nodes.Getattr):
        return _is_var_access(node.node, varname) and node.attr == key
    return False


def _is_var_or_elems_access(node: jinja2.nodes.Node, varname: str, key: str | None = None) -> bool:
    if isinstance(node, jinja2.nodes.Filter):
        return node.node is not None and _is_var_or_elems_access(node.node, varname, key)
    if isinstance(node, jinja2.nodes.Test):
        return _is_var_or_elems_access(node.node, varname, key)
    if isinstance(node, jinja2.nodes.Getitem) and isinstance(node.arg, jinja2.nodes.Slice):
        return _is_var_or_elems_access(node.node, varname, key)
    return _is_attr_access(node, varname, key) if key else _is_var_access(node, varname)


def _iter_nodes_assign_var_or_elems(root: jinja2.nodes.Node, varname: str) -> Iterator[tuple[jinja2.nodes.Node, str]]:
    # The variable itself, defined at the root
    yield root, varname
    # Then, breadth first, every variable assigned from it or from one of those
    related_varnames = deque([varname])
    while related_varnames:
        related_varname = related_varnames.popleft()
        for assign_ast in root.find_all(jinja2.nodes.Assign):
            lhs = assign_ast.target
            rhs = assign_ast.node
            if _is_var_or_elems_access(rhs, related_varname):
                assert isinstance(lhs, jinja2.nodes.Name)
                yield assign_ast, lhs.name
                # Avoid looping forever on a self-assignment
                if lhs.name != related_varname:
                    related_varnames.append(lhs.name)


def _iter_nodes_assign_messages_item(root: jinja2.nodes.Node) -> Iterator[tuple[jinja2.nodes.Node, str]]:
    messages_varnames = [varname for _, varname in _iter_nodes_assign_var_or_elems(root, "messages")]
    # {%- for message in messages -%} loops
    for loop_ast in root.find_all(jinja2.nodes.For):
        loop_iter = loop_ast.iter
        loop_target = loop_ast.target
        for varname in messages_varnames:
            if _is_var_or_elems_access(loop_iter, varname):
                assert isinstance(loop_target, jinja2.nodes.Name)
                yield loop_ast, loop_target.name
                break


def _loops_over_content(root: jinja2.nodes.Node) -> Iterator[tuple[jinja2.nodes.Node, str]]:
    """The loops over a message's content, or over a macro parameter bound to one: the engine's
    ``_iter_nodes_assign_content_item``."""
    message_varnames = [varname for _, varname in _iter_nodes_assign_messages_item(root)]
    # Macro parameters that receive a message's content: some templates pass message.content through a parameter whose
    # name is not "content".
    macro_content_params_by_loop: dict[int, set[str]] = {}
    loops_in_macros: set[int] = set()
    for macro_node in root.find_all(jinja2.nodes.Macro):
        macro_param_names = {arg.name for arg in macro_node.args}
        macro_content_params: set[str] = set()
        for call_node in root.find_all(jinja2.nodes.Call):
            if isinstance(call_node.node, jinja2.nodes.Name) and call_node.node.name == macro_node.name:
                for i, arg in enumerate(call_node.args):
                    if i < len(macro_node.args) and any(
                        _is_var_or_elems_access(arg, varname, "content") for varname in message_varnames
                    ):
                        macro_content_params.add(macro_node.args[i].name)
                for kwarg in call_node.kwargs:
                    if (
                        isinstance(kwarg, jinja2.nodes.Keyword)
                        and kwarg.key in macro_param_names
                        and any(
                            _is_var_or_elems_access(kwarg.value, varname, "content") for varname in message_varnames
                        )
                    ):
                        macro_content_params.add(kwarg.key)
        for loop_ast in macro_node.find_all(jinja2.nodes.For):
            loops_in_macros.add(id(loop_ast))
            if macro_content_params:
                macro_content_params_by_loop[id(loop_ast)] = macro_content_params
    # {%- for content in message['content'] -%} loops, or {%- for item in content -%} loops
    for loop_ast in root.find_all(jinja2.nodes.For):
        loop_iter = loop_ast.iter
        loop_target = loop_ast.target
        for varname in message_varnames:
            if _is_var_or_elems_access(loop_iter, varname, "content"):
                assert isinstance(loop_target, jinja2.nodes.Name)
                yield loop_ast, loop_target.name
                break
        else:
            macro_content_params_for_loop = macro_content_params_by_loop.get(id(loop_ast))
            if (
                isinstance(loop_iter, jinja2.nodes.Name)
                and macro_content_params_for_loop is not None
                and loop_iter.name in macro_content_params_for_loop
            ):
                assert isinstance(loop_target, jinja2.nodes.Name)
                yield loop_ast, loop_target.name
                continue
            if (
                id(loop_ast) not in loops_in_macros
                and isinstance(loop_iter, jinja2.nodes.Name)
                and loop_iter.name == "content"
            ):
                assert isinstance(loop_target, jinja2.nodes.Name)
                yield loop_ast, loop_target.name
