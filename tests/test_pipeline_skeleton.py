import importlib


def test_imports():
    # Ensure core modules import without heavy optional dependencies
    assert importlib.import_module("src.preprocessing")
    assert importlib.import_module("src.layout")
    assert importlib.import_module("src.correction")
    assert importlib.import_module("src.structuring")
    assert importlib.import_module("src.schema")


def test_cli_parsing():
    from src.orchestrator import parse_args
    ns = parse_args(["--input", "sample.pdf", "--output", "out.json", "--dry-run", "--no-embeddings", "--layout-strategy", "fallback"])
    assert ns.input.endswith("sample.pdf")
    assert ns.output.endswith("out.json")
    assert ns.dry_run is True
    assert ns.no_embeddings is True
