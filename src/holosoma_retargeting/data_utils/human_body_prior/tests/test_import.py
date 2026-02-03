def test_package_imports():
    import importlib
    mod = importlib.import_module("human_body_prior")
    assert mod is not None

