import pytest

from ueoa.orchestrator.expressions import (
    ExpressionError,
    MissingValueError,
    evaluate_source,
    parse,
    references,
    resolve_bindings,
)

CONTEXT = {
    "inputs": {"contrast_keys": ["key_a"], "profile": "profile/standard@1"},
    "nodes": {
        "check": {"output": {"ready": True, "results": [{"id": "r1"}, {"id": "r2"}]}},
        "analyze": {"items": [{"key": "d1", "output": {"n": 1}}, {"key": "d2", "output": {"n": 2}}]},
    },
    "run": {"locale": "de-DE", "id": "run-1"},
    "item": {"key": "d1", "payload": {"title": "x"}},
}


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("len($inputs.contrast_keys) > 0", True),
        ("$nodes.check.output.ready == true", True),
        ("$nodes.analyze.items[*].output.n", [1, 2]),
        ("$nodes.analyze.items[1].key", "d2"),
        ("$nodes.contrast.output | default(null)", None),
        ("$nodes.missing.output | default([])", []),
        ("exists($nodes.contrast.output)", False),
        ("exists($run.locale)", True),
        ("not ($run.locale == 'en') and len($nodes.check.output.results) >= 2", True),
        ("false or 1 < 2", True),
        ("$item.payload", {"title": "x"}),
        ("['a', \"b\"]", ["a", "b"]),
        ("2.5 <= 3", True),
    ],
)
def test_evaluate(source: str, expected: object) -> None:
    assert evaluate_source(source, CONTEXT) == expected


@pytest.mark.parametrize(
    "source",
    [
        "__import__('os').system('x')",
        "$nodes.check.output.ready; drop",
        "$env.SECRET",
        "len($inputs.x",
        "",
        "1 +",
        "$inputs.profile | upper()",
        "open('x')",
    ],
)
def test_rejects_invalid_or_unsafe_syntax(source: str) -> None:
    with pytest.raises(ExpressionError):
        parse(source)


def test_flatten() -> None:
    context = {"nodes": {"a": {"items": [{"output": {"r": [1, 2]}}, {"output": {"r": []}}, {"output": {"r": [3]}}]}}}
    assert evaluate_source("flatten($nodes.a.items[*].output.r)", context) == [1, 2, 3]
    assert evaluate_source("flatten([[1], 2, [[3]]])", {}) == [1, 2, [3]]
    assert evaluate_source("len(flatten($nodes.a.items[*].output.r)) == 3", context) is True
    assert [use.ref.text() for use in references(parse("flatten($nodes.a.items[*].output.r)"))] == [
        "$nodes.a.items[*].output.r"
    ]
    with pytest.raises(ExpressionError, match="flatten\\(\\) needs a list"):
        evaluate_source("flatten('x')", {})


def test_missing_reference_raises() -> None:
    with pytest.raises(MissingValueError):
        evaluate_source("$nodes.contrast.output", CONTEXT)


def test_type_errors_are_expression_errors() -> None:
    with pytest.raises(ExpressionError):
        evaluate_source("$run.locale > 3", CONTEXT)
    with pytest.raises(ExpressionError):
        evaluate_source("$run.locale and true", CONTEXT)


def test_reference_guarding() -> None:
    uses = references(parse("len($nodes.a.output) > 0 and exists($nodes.b.x) or ($nodes.c.y | default(1)) > 0"))
    guarded = {use.ref.text(): use.guarded for use in uses}
    assert guarded == {"$nodes.a.output": False, "$nodes.b.x": True, "$nodes.c.y": True}


def test_resolve_bindings_keeps_literals() -> None:
    bindings = {"profile": "$inputs.profile", "fixed": "profile/x@1", "nested": {"l": ["$run.locale", 3]}}
    assert resolve_bindings(bindings, CONTEXT) == {
        "profile": "profile/standard@1",
        "fixed": "profile/x@1",
        "nested": {"l": ["de-DE", 3]},
    }


def test_expression_length_limit() -> None:
    with pytest.raises(ExpressionError):
        parse("$run.id == '" + "a" * 2000 + "'")
