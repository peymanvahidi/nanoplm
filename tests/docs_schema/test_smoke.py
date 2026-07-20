def test_package_importable():
    import tools.schema_gen as sg
    assert sg.__name__ == "tools.schema_gen"
