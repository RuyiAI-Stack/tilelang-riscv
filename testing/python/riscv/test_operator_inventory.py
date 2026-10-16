"""Guard coverage accounting against upstream schema and status changes."""

from collections import Counter
import xml.etree.ElementTree as ET

import pytest

from operator_inventory import Declarations, attach_results, inventories, read_manifest, validate


def test_current_manifest_and_factory_inventory():
    libraries = inventories()
    validate(libraries)
    ops = libraries["TileOps"]["entries"]
    assert len(ops) == 205
    assert Counter(e["upstream_status"] for e in ops) == {"implemented": 191, "spec-only": 14}
    assert all(e["name"] != "adts" for e in ops)
    paged = next(e for e in ops if e["name"] == "GQAPagedFwdOp")
    assert paged["upstream_status"] == "spec-only"
    assert paged["kernel_classes"] and paged["tests"]
    cache_write = next(e for e in ops if e["name"] == "PagedKVCacheWriteFwdOp")
    assert cache_write["upstream_status"] == "implemented"
    assert cache_write["kernel_classes"] and cache_write["tests"]
    cache_gather = next(e for e in ops if e["name"] == "PagedKVCacheGatherFwdOp")
    assert cache_gather["upstream_status"] == "implemented"
    assert cache_gather["kernel_classes"] and cache_gather["tests"]
    assert len(libraries["TileKernels"]["entries"]) == 48


def test_missing_or_changed_manifest_path_cannot_report_empty_success(tmp_path):
    with pytest.raises(ValueError, match="No TileOps operators"):
        read_manifest(tmp_path)


def test_resolves_inherited_mappings_and_delegate_expansion(tmp_path):
    (tmp_path / "ops.py").write_text(
        "class Base:\n    kernel_types = {'basic': BasicKernel}\n"
        "class Child(Base):\n    pass\n"
        "class Composite:\n    delegate_types = {'child': Child}\n"
        "class Extended(Composite):\n    delegate_types = {**Composite.delegate_types, 'base': Base}\n"
    )
    declarations = Declarations(tmp_path)
    assert declarations.mapping("Child", "kernel_types") == {"basic": "BasicKernel"}
    assert declarations.mapping("Extended", "delegate_types") == {"child": "Child", "base": "Base"}
    assert declarations.kernels("Extended") == {"BasicKernel"}


def test_junit_rejections_and_expected_failures_are_not_numerical_passes(tmp_path):
    suite = ET.Element("testsuite")
    for name, status in (
        ("test_value", "passed"),
        ("test_value_bug", "xfail"),
        ("test_x_rejects_cuda_extern", "passed"),
        ("test_x_lowers_to_riscv", "passed"),
    ):
        case = ET.SubElement(suite, "testcase", classname="testing.python.riscv.TileOps.test_example", name=name)
        if status == "xfail":
            ET.SubElement(case, "skipped", type="pytest.xfail")
    path = tmp_path / "results.xml"
    ET.ElementTree(suite).write(path)
    entry = dict(name="ExampleOp", tests=["test_example.py"], upstream_status="spec-only")
    libraries = {"TileOps": {"entries": [entry]}}
    attach_results(libraries, path)
    assert entry["upstream_status"] == "spec-only"
    assert entry["tests_outcomes"] == {"passed": 3, "expected-failure": 1}
    assert entry["tests_passed_scopes"] == {"kernel-or-component-numerical": 1, "rejection": 1, "compile-only": 1}
