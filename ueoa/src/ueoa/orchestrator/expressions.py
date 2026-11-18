"""Restricted, side-effect-free expression language for WorkflowSpec conditions and bindings.

Grammar (parsed and evaluated in code, never via eval):
    expr       := or_expr
    or_expr    := and_expr ("or" and_expr)*
    and_expr   := not_expr ("and" not_expr)*
    not_expr   := "not" not_expr | comparison
    comparison := pipe (("==" | "!=" | ">" | ">=" | "<" | "<=") pipe)?
    pipe       := primary ("|" "default" "(" expr ")")*
    primary    := NUMBER | STRING | "true" | "false" | "null" | REF
                | "[" [expr ("," expr)*] "]" | ("len" | "exists" | "flatten") "(" expr ")" | "(" expr ")"
    REF        := "$" root ("." name | "[*]" | "[" int "]")*
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

ROOTS = frozenset({"inputs", "nodes", "item", "run", "node"})
FUNCTIONS = frozenset({"len", "exists", "flatten"})
KEYWORDS = frozenset({"and", "or", "not", "true", "false", "null", "default"}) | FUNCTIONS
MAX_EXPRESSION_LENGTH = 1000

WILDCARD = "*"


class ExpressionError(ValueError):
    """Raised for syntax errors and invalid evaluation."""


class MissingValueError(ExpressionError):
    """Raised when a reference cannot be resolved (missing key, skipped node, ...)."""


# ---------------------------------------------------------------- AST


@dataclass(frozen=True)
class Literal:
    value: Any


@dataclass(frozen=True)
class Ref:
    root: str
    path: tuple[str | int, ...]

    def text(self) -> str:
        parts = [f"${self.root}"]
        for segment in self.path:
            if segment == WILDCARD:
                parts.append("[*]")
            elif isinstance(segment, int):
                parts.append(f"[{segment}]")
            else:
                parts.append(f".{segment}")
        return "".join(parts)


@dataclass(frozen=True)
class ListLit:
    items: tuple[Node, ...]


@dataclass(frozen=True)
class Call:
    name: str
    arg: Node


@dataclass(frozen=True)
class Default:
    expr: Node
    fallback: Node


@dataclass(frozen=True)
class Not:
    expr: Node


@dataclass(frozen=True)
class BoolOp:
    op: str
    left: Node
    right: Node


@dataclass(frozen=True)
class Compare:
    op: str
    left: Node
    right: Node


Node = Literal | Ref | ListLit | Call | Default | Not | BoolOp | Compare


# ---------------------------------------------------------------- lexer

_TOKEN_RE = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<number>-?\d+(?:\.\d+)?)
  | (?P<string>'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")
  | (?P<ref>\$[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*|\[\*\]|\[\d+\])*)
  | (?P<op>==|!=|>=|<=|>|<)
  | (?P<ident>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<punct>[()\[\],|])
    """,
    re.VERBOSE,
)

_REF_SEGMENT_RE = re.compile(r"\.([A-Za-z_][A-Za-z0-9_]*)|\[(\*|\d+)\]")


@dataclass(frozen=True)
class _Token:
    kind: str
    text: str
    pos: int


def _tokenize(source: str) -> list[_Token]:
    tokens: list[_Token] = []
    pos = 0
    while pos < len(source):
        match = _TOKEN_RE.match(source, pos)
        if match is None:
            raise ExpressionError(f"unexpected character {source[pos]!r} at {pos}")
        kind = match.lastgroup or ""
        if kind != "ws":
            tokens.append(_Token(kind, match.group(), pos))
        pos = match.end()
    return tokens


def _parse_ref(text: str) -> Ref:
    head = re.match(r"\$([A-Za-z_][A-Za-z0-9_]*)", text)
    if head is None:  # pragma: no cover - guaranteed by the lexer
        raise ExpressionError(f"invalid reference {text!r}")
    root = head.group(1)
    if root not in ROOTS:
        raise ExpressionError(f"unknown reference root ${root}; allowed: {sorted(ROOTS)}")
    path: list[str | int] = []
    for segment in _REF_SEGMENT_RE.finditer(text, head.end()):
        name, index = segment.group(1), segment.group(2)
        if name is not None:
            path.append(name)
        elif index == WILDCARD:
            path.append(WILDCARD)
        else:
            path.append(int(index))
    return Ref(root, tuple(path))


def _unquote(text: str) -> str:
    body = text[1:-1]
    return re.sub(r"\\(.)", r"\1", body)


# ---------------------------------------------------------------- parser


class _Parser:
    def __init__(self, source: str) -> None:
        self.tokens = _tokenize(source)
        self.pos = 0

    def parse(self) -> Node:
        node = self._or()
        if self.pos != len(self.tokens):
            token = self.tokens[self.pos]
            raise ExpressionError(f"unexpected {token.text!r} at {token.pos}")
        return node

    def _peek(self) -> _Token | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _accept(self, text: str) -> bool:
        token = self._peek()
        if token is not None and token.text == text and token.kind in {"ident", "op", "punct"}:
            self.pos += 1
            return True
        return False

    def _expect(self, text: str) -> None:
        if not self._accept(text):
            token = self._peek()
            found = "end of expression" if token is None else repr(token.text)
            raise ExpressionError(f"expected {text!r}, found {found}")

    def _or(self) -> Node:
        node = self._and()
        while self._accept("or"):
            node = BoolOp("or", node, self._and())
        return node

    def _and(self) -> Node:
        node = self._not()
        while self._accept("and"):
            node = BoolOp("and", node, self._not())
        return node

    def _not(self) -> Node:
        if self._accept("not"):
            return Not(self._not())
        return self._comparison()

    def _comparison(self) -> Node:
        left = self._pipe()
        token = self._peek()
        if token is not None and token.kind == "op":
            self.pos += 1
            return Compare(token.text, left, self._pipe())
        return left

    def _pipe(self) -> Node:
        node = self._primary()
        while self._accept("|"):
            self._expect("default")
            self._expect("(")
            fallback = self._or()
            self._expect(")")
            node = Default(node, fallback)
        return node

    def _primary(self) -> Node:
        token = self._peek()
        if token is None:
            raise ExpressionError("unexpected end of expression")
        self.pos += 1
        if token.kind == "number":
            return Literal(float(token.text) if "." in token.text else int(token.text))
        if token.kind == "string":
            return Literal(_unquote(token.text))
        if token.kind == "ref":
            return _parse_ref(token.text)
        if token.kind == "ident":
            if token.text == "true":
                return Literal(True)
            if token.text == "false":
                return Literal(False)
            if token.text == "null":
                return Literal(None)
            if token.text in FUNCTIONS:
                self._expect("(")
                arg = self._or()
                self._expect(")")
                return Call(token.text, arg)
            raise ExpressionError(f"unknown identifier {token.text!r} at {token.pos}")
        if token.text == "(":
            node = self._or()
            self._expect(")")
            return node
        if token.text == "[":
            items: list[Node] = []
            if not self._accept("]"):
                items.append(self._or())
                while self._accept(","):
                    items.append(self._or())
                self._expect("]")
            return ListLit(tuple(items))
        raise ExpressionError(f"unexpected {token.text!r} at {token.pos}")


def parse(source: str) -> Node:
    if len(source) > MAX_EXPRESSION_LENGTH:
        raise ExpressionError(f"expression longer than {MAX_EXPRESSION_LENGTH} characters")
    if not source.strip():
        raise ExpressionError("empty expression")
    return _Parser(source).parse()


def is_expression(value: object) -> bool:
    """Binding values are expressions only when they are strings starting with '$'."""
    return isinstance(value, str) and value.lstrip().startswith("$")


# ---------------------------------------------------------------- static analysis


@dataclass(frozen=True)
class RefUse:
    ref: Ref
    guarded: bool  # inside default(...) or exists(...): a missing value is tolerated


@dataclass
class _Collector:
    uses: list[RefUse] = field(default_factory=lambda: [])

    def visit(self, node: Node, guarded: bool) -> None:
        match node:
            case Ref():
                self.uses.append(RefUse(node, guarded))
            case Literal():
                pass
            case ListLit(items=items):
                for item in items:
                    self.visit(item, guarded)
            case Call(name=name, arg=arg):
                self.visit(arg, guarded or name == "exists")
            case Default(expr=expr, fallback=fallback):
                self.visit(expr, True)
                self.visit(fallback, guarded)
            case Not(expr=expr):
                self.visit(expr, guarded)
            case BoolOp(left=left, right=right) | Compare(left=left, right=right):
                self.visit(left, guarded)
                self.visit(right, guarded)


def references(node: Node) -> list[RefUse]:
    collector = _Collector()
    collector.visit(node, guarded=False)
    return collector.uses


# ---------------------------------------------------------------- evaluation


def _walk(value: Any, path: tuple[str | int, ...], text: str) -> Any:
    current = value
    for index, segment in enumerate(path):
        if segment == WILDCARD:
            if not isinstance(current, list):
                raise MissingValueError(f"{text}: [*] applied to non-list")
            rest = path[index + 1 :]
            return [_walk(element, rest, text) for element in current]  # pyright: ignore[reportUnknownVariableType]
        if isinstance(segment, int):
            if not isinstance(current, list) or segment >= len(current):  # pyright: ignore[reportUnknownArgumentType]
                raise MissingValueError(f"{text}: index {segment} not available")
            current = current[segment]
        else:
            if not isinstance(current, dict) or segment not in current:
                raise MissingValueError(f"{text}: '{segment}' not available")
            current = current[segment]  # pyright: ignore[reportUnknownVariableType]
    return current  # pyright: ignore[reportUnknownVariableType]


def _compare(op: str, left: Any, right: Any) -> bool:
    try:
        match op:
            case "==":
                return bool(left == right)
            case "!=":
                return bool(left != right)
            case ">":
                return bool(left > right)
            case ">=":
                return bool(left >= right)
            case "<":
                return bool(left < right)
            case "<=":
                return bool(left <= right)
    except TypeError as exc:
        raise ExpressionError(f"cannot compare {type(left).__name__} {op} {type(right).__name__}") from exc
    raise ExpressionError(f"unknown operator {op}")  # pragma: no cover


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    raise ExpressionError(f"expected a boolean, got {type(value).__name__}")


def evaluate(node: Node, context: dict[str, Any]) -> Any:
    match node:
        case Literal(value=value):
            return value
        case Ref(root=root, path=path):
            if root not in context:
                raise MissingValueError(f"{node.text()}: ${root} not available here")
            return _walk(context[root], path, node.text())
        case ListLit(items=items):
            return [evaluate(item, context) for item in items]
        case Default(expr=expr, fallback=fallback):
            try:
                value = evaluate(expr, context)
            except MissingValueError:
                return evaluate(fallback, context)
            return evaluate(fallback, context) if value is None else value
        case Call(name="exists", arg=arg):
            try:
                return evaluate(arg, context) is not None
            except MissingValueError:
                return False
        case Call(name="len", arg=arg):
            value = evaluate(arg, context)
            if not isinstance(value, list | dict | str):
                raise ExpressionError(f"len() needs a list, object or string, got {type(value).__name__}")
            return len(value)  # pyright: ignore[reportUnknownArgumentType]
        case Call(name="flatten", arg=arg):
            value = evaluate(arg, context)
            if not isinstance(value, list):
                raise ExpressionError(f"flatten() needs a list, got {type(value).__name__}")
            flat: list[Any] = []
            for element in value:  # pyright: ignore[reportUnknownVariableType]
                if isinstance(element, list):
                    flat.extend(element)  # pyright: ignore[reportUnknownArgumentType]
                else:
                    flat.append(element)
            return flat
        case Not(expr=expr):
            return not _truthy(evaluate(expr, context))
        case BoolOp(op="and", left=left, right=right):
            return _truthy(evaluate(left, context)) and _truthy(evaluate(right, context))
        case BoolOp(op="or", left=left, right=right):
            return _truthy(evaluate(left, context)) or _truthy(evaluate(right, context))
        case Compare(op=op, left=left, right=right):
            return _compare(op, evaluate(left, context), evaluate(right, context))
        case _:  # pragma: no cover
            raise ExpressionError(f"unsupported node {node!r}")


def evaluate_source(source: str, context: dict[str, Any]) -> Any:
    return evaluate(parse(source), context)


def resolve_bindings(value: Any, context: dict[str, Any]) -> Any:
    """Resolve node input bindings: '$...' strings are expressions, everything else is a literal."""
    if is_expression(value):
        return evaluate_source(str(value), context)
    if isinstance(value, dict):
        return {key: resolve_bindings(item, context) for key, item in value.items()}  # pyright: ignore[reportUnknownVariableType]
    if isinstance(value, list):
        return [resolve_bindings(item, context) for item in value]  # pyright: ignore[reportUnknownVariableType]
    return value
