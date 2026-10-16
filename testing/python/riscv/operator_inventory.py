"""Index pinned upstream declarations and local test references without importing CUDA.

References are navigation aids, not claims of full operator support. Manifest
status, declared kernels, test scope, and observed JUnit outcomes stay separate.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

import yaml

ROOT = Path(__file__).resolve().parents[3]
TEST_ROOT = Path(__file__).resolve().parent

# Factories accessed directly need references beyond the public Kernel class.
# A selector narrows mixed test modules to the relevant test functions.
EXPLICIT_TESTS = {
    "MLADecodeWithKVCacheFwdOp": ["test_upstream_attention.py::test_mla_mma_runtime", "test_mla_decode.py"],
    "GQAVarlenFwdOp": ["test_upstream_attention.py::test_varlen_upstream_runtime"],
    "GQAPagedFwdOp": ["test_upstream_attention.py::test_paged_upstream_runtime", "test_gqa_paged_features.py"],
    "GDNFwdOp": ["test_upstream_gdn.py"],
    "MoEGroupedGemmFwdOp": ["test_moe_grouped_gemm.py"],
    "GLAFwdOp": ["test_gla_fwd.py"],
    "GLAChunkFwdOp": ["test_gla_fwd.py"],
    "MHADecodePagedWithKVCacheFwdOp": ["test_mha_decode_paged.py"],
    "IndexedExpertMLPFwdOp": ["test_indexed_expert_mlp.py"],
    "SharedExpertMLPFwdOp": ["test_moe_shared_expert_mlp.py"],
    "DSADecodeWithKVCacheFwdOp": ["test_sparse_mla.py"],
    "GemmFwdOp": ["test_gemm.py::test_gemm_basic_runtime_compare"],
    "GemmW4A16FwdOp": ["test_gemm.py::test_gemm_w4a16_mma_runtime_compare", "test_gemm.py::test_gemm_w4a16_rejects_cuda_dequant_extern"],
    "TopKMaskFwdOp": ["test_sampling_masks.py::test_top_k_mask_ragged_runtime_compare"],
    "TopPMaskFwdOp": ["test_sampling_masks.py::test_top_p_mask_ragged_runtime_compare"],
    "TopKTopPMaskFwdOp": ["test_sampling_masks.py::test_top_k_top_p_mask_ragged_runtime_compare"],
    "INT8DequantPerChannelFwdOp": ["test_int8_dequant.py::test_int8_dequant_per_channel_runtime_compare"],
    "INT8DequantPerBlockFwdOp": ["test_int8_dequant.py::test_int8_dequant_per_block_runtime_compare"],
    "PagedKVCacheWriteFwdOp": ["test_paged_kv_cache_write.py"],
    "PagedKVCacheGatherFwdOp": ["test_paged_cache_gather.py"],
    "DeltaNetFwdOp": ["test_delta_decode.py::test_deltanet_decode_runtime_compare"],
    "KDAFwdOp": ["test_delta_decode.py::test_kda_decode_runtime_compare"],
    "ArgmaxFwdOp": ["test_reduction.py::test_argreduce_float32_runtime_compare[argmax-"],
    "ArgminFwdOp": ["test_reduction.py::test_argreduce_float32_runtime_compare[argmin-"],
}
COMPONENT_ONLY = {"GLAFwdOp", "GLAChunkFwdOp", "IndexedExpertMLPFwdOp", "SharedExpertMLPFwdOp"}
# These tests select generalized upstream factories by operation kind.
for _name, _function, _parameter in (
    ("Sum", "reduce", "sum"),
    ("Mean", "reduce", "mean"),
    ("Amax", "reduce", "amax"),
    ("Amin", "reduce", "amin"),
    ("Prod", "reduce", "prod"),
    ("Std", "reduce", "std"),
    ("Var", "reduce", "var"),
    ("Softmax", "softmax", "softmax"),
    ("LogSoftmax", "softmax", "log_softmax"),
    ("Any", "logical_reduce", "any"),
    ("All", "logical_reduce", "all"),
    ("CountNonzero", "logical_reduce", "count_nonzero"),
    ("Cumsum", "cumulative", "sum"),
    ("Cumprod", "cumulative", "prod"),
):
    EXPLICIT_TESTS[_name + "FwdOp"] = [f"test_reduction.py::test_{_function}_float32_runtime_compare[{_parameter}"]
EXPLICIT_TESTS.update(
    {
        "VarMeanFwdOp": ["test_reduction.py::test_reduce_var_mean_float32_runtime_compare"],
        "VectorNormFwdOp": ["test_reduction.py::test_vector_norm_float32_runtime_compare"],
        "LogSumExpFwdOp": [
            "test_reduction.py::test_logsumexp_float32_runtime_compare",
            "test_reduction.py::test_logsumexp_tiled_float32_runtime_compare",
        ],
    }
)
for _dimension in (1, 3):
    for _indices in (False, True):
        _name = f"MaxPool{_dimension}d" + ("Indices" if _indices else "") + "FwdOp"
        EXPLICIT_TESTS[_name] = [f"test_pool_extended.py::test_max_pool_runtime_compare[{_dimension}-{_indices}]"]
PUBLIC_API_TESTS = {
    "GQAVarlenFwdOp": ["test_upstream_public_api.py::test_public_varlen_dispatch_to_upstream"],
    "PagedKVCacheGatherFwdOp": ["test_upstream_public_api.py::test_public_gather_preserves_strided_destination"],
}
KNOWN_GAPS = {
    "GQAPagedFwdOp": [
        "General FP16/BF16 programs have selected RoPE/window/sink and split numerical coverage; public dispatch and WS RoPE preparation are unvalidated",
        "Upstream paged GQA still requires an external implementation for FP8",
    ],
    "SigmoidFwdOp": [
        "FP32 generic-load numerics are tested; aligned register-copy uses CUDA-only tileops::load16_evict_last",
        "FP16/BF16 sigmoid uses unsupported CUDA __fdividef, including the generic-load path",
    ],
    "MLADecodeWithKVCacheFwdOp": ["WS variant calls unsupported tileops::uniform_wgmma_bases; MMA numerics are tested"],
    "GemmW4A16FwdOp": ["WGMMA variant calls unsupported tileops::w4a16_dequant_word; original MMA FP16/BF16 numerics are tested"],
    "FP8QuantPerBlockFwdOp": ["Unaligned FP16/BF16/FP32 numerics are tested; aligned variant requires tileops::load16 helpers"],
    "RMSNormFwdOp": [
        "Ragged FP32 width uses the original generic-load path; aligned/warp-row vector fast paths call tileops::load16_evict_last"
    ],
    "SoftmaxFwdOp": ["Ragged 513-column FP32 row uses generic loads; aligned held-row fast path calls tileops::load16_evict_last"],
    "INT8QuantPerTensorFwdOp": ["Whole vectors require tileops::load16 helpers; multiple CTAs also require grid-wide synchronization"],
    "INT8QuantPerChannelFwdOp": ["Whole vectors require tileops::load16 helpers; only short scalar-tail probes have run"],
    "INT8QuantPerBlockFwdOp": ["Whole vectors require tileops::load16_evict_first; only short scalar-tail probes have run"],
    "INT4QuantPerGroupFwdOp": ["First blocker is uint32-to-float16x2 reinterpret; also requires CUDA-specific tileops::load16 helpers"],
    "SmoothQuantFwdOp": ["Whole vectors require tileops::load16 helpers"],
    "INT8DequantPerTensorFwdOp": ["Calls CUDA-specific tileops::load_f32_evict_last"],
    "MoEGroupedGemmFP8FwdOp": [
        "Minimal original-kernel probe fails TIR opaque-access region inference before codegen; source also uses SM90 TMA/WGMMA and tileops::moe_fp8 helpers"
    ],
    "TopKMaskFwdOp": ["Numerics cover FP32 scalar loads; packed 16-bit and clustered variants are not validated"],
    "TopPMaskFwdOp": ["Numerics cover FP32 scalar loads and one CTA per row; multi-CTA grid synchronization is not supported"],
    "TopKTopPMaskFwdOp": ["Numerics cover FP32 scalar loads; packed 16-bit and clustered variants are not validated"],
    "SamplingFromProbsFwdOp": [
        "Numerics cover scalar loads and one CTA per row; vector loads and split-row grid synchronization remain unvalidated"
    ],
    "ChainSpeculativeSamplingFwdOp": [
        "Numerics cover scalar loads and one CTA per row; vector loads and split-row grid synchronization remain unvalidated"
    ],
    "DeltaNetFwdOp": ["Inference decode FP16/BF16 numerics are tested; full inference prefill/varlen composition is not validated"],
    "KDAFwdOp": [
        "Decode FP16/BF16 numerics are tested; default 256-thread chunk-prefill overflows the host stack and 32-thread prepare/scan has incorrect numerics"
    ],
}


def revision(path: Path) -> str | None:
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def references(name: str, tests: dict[str, str]) -> list[str]:
    pattern = re.compile(r"\b" + re.escape(name) + r"\b")
    return sorted(file for file, text in tests.items() if pattern.search(text))


def read_manifest(source: Path) -> list[tuple[str, dict, Path]]:
    entries = []
    for path in sorted((source / "src/tileops/manifest/spec").glob("*.yaml")):
        if path.name == "types.yaml":
            continue
        data = yaml.safe_load(path.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"Invalid operator manifest: {path}")
        for name, spec in data.items():
            if not isinstance(spec, dict) or not name.endswith("Op") or "status" not in spec:
                raise ValueError(f"Unexpected operator declaration: {path}:{name}")
            entries.append((name, spec, path))
    if not entries:
        raise ValueError(f"No TileOps operators found under {source}; check the manifest schema/path")
    names = [name for name, _, _ in entries]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate TileOps manifest operator names")
    return entries


class Declarations:
    """Resolve class mappings, inheritance and dictionary expansion via source AST."""

    def __init__(self, path: Path):
        self.classes = {}
        for file in sorted(path.rglob("*.py")):
            for node in ast.parse(file.read_text()).body:
                if isinstance(node, ast.ClassDef):
                    if node.name in self.classes:
                        raise ValueError(f"Ambiguous class name: {node.name}")
                    self.classes[node.name] = (node, file)

    def mapping(self, name: str, attribute: str, seen: frozenset = frozenset()) -> dict[str, str]:
        if name not in self.classes:
            return {}
        key = (name, attribute)
        if key in seen:
            raise ValueError(f"Cyclic class mapping: {key}")
        seen = seen | {key}
        node, _ = self.classes[name]
        for stmt in node.body:
            targets = [stmt.target] if isinstance(stmt, ast.AnnAssign) else stmt.targets if isinstance(stmt, ast.Assign) else []
            if any(isinstance(t, ast.Name) and t.id == attribute for t in targets):
                value = stmt.value
                if isinstance(value, ast.Call) and ast.unparse(value.func) == "MappingProxyType":
                    value = value.args[0]
                if not isinstance(value, ast.Dict):
                    raise ValueError(f"Unsupported mapping expression: {name}.{attribute}")
                result = {}
                for k, v in zip(value.keys, value.values):
                    if k is None and isinstance(v, ast.Attribute) and isinstance(v.value, ast.Name):
                        result.update(self.mapping(v.value.id, v.attr, seen))
                    elif isinstance(k, ast.Constant):
                        result[str(k.value)] = ast.unparse(v)
                    elif isinstance(k, ast.Name):
                        result[k.id] = ast.unparse(v)
                    else:
                        raise ValueError(f"Unsupported mapping entry: {name}.{attribute}")
                return result
        # Follow normal left-to-right inheritance; current upstream has one
        # mapping-bearing base per class. Empty mappings on Op are explicit.
        for base in node.bases:
            if isinstance(base, ast.Name):
                result = self.mapping(base.id, attribute, seen)
                if result:
                    return result
        return {}

    def kernels(self, name: str, seen: frozenset = frozenset()) -> set[str]:
        if name in seen:
            raise ValueError(f"Cyclic Op delegation: {name}")
        result = set(self.mapping(name, "kernel_types").values())
        for delegate in self.mapping(name, "delegate_types").values():
            result.update(self.kernels(delegate, seen | {name}))
        return result


def inventories(root: Path = ROOT, test_root: Path = TEST_ROOT) -> dict:
    libraries = {}
    for library, directory in (("TileOps", "TileOPs"), ("TileKernels", "TileKernels")):
        source = root / "3rdparty" / directory
        tests = {p.name: p.read_text() for p in (test_root / library).glob("test_*.py")}
        entries = []
        if library == "TileOps":
            declarations = Declarations(source / "src/tileops/ops")
            manifest = read_manifest(source)
            manifest_names = {name for name, _, _ in manifest}
            stale = (EXPLICIT_TESTS.keys() | PUBLIC_API_TESTS.keys() | KNOWN_GAPS.keys()) - manifest_names
            if stale:
                raise ValueError(f"Stale explicit operator mappings: {sorted(stale)}")
            for name, spec, path in manifest:
                kernels = sorted(declarations.kernels(name))
                delegates = declarations.mapping(name, "delegate_types")
                files = sorted({f for k in kernels for f in references(k, tests)})
                explicit = EXPLICIT_TESTS.get(name, [])
                # Prefer precise selectors when present for a mixed test file.
                explicit_files = {r.split("::")[0] for r in explicit}
                files = sorted(set(f for f in files if f not in explicit_files) | set(explicit))
                node = declarations.classes.get(name)
                entries.append(
                    dict(
                        name=name,
                        family=spec["family"],
                        upstream_status=spec["status"],
                        manifest=str(path.relative_to(root)),
                        op_source=str(node[1].relative_to(root)) if node else None,
                        kernel_classes=kernels,
                        interfaces=declarations.mapping(name, "interfaces"),
                        delegates=delegates,
                        evidence="component-test-references"
                        if files and (delegates or name in COMPONENT_ONLY)
                        else "kernel-test-references"
                        if files
                        else "unmapped",
                        tests=files,
                        public_api_tests=PUBLIC_API_TESTS.get(name, []),
                        known_gaps=KNOWN_GAPS.get(name, []),
                    )
                )
            by_name = {entry["name"]: entry for entry in entries}

            def inherit_test_references(entry, seen=frozenset(), by_name=by_name):
                if entry["name"] in seen:
                    raise ValueError(f"Cyclic manifest delegation: {entry['name']}")
                for delegate in entry["delegates"].values():
                    if delegate in by_name:
                        child = by_name[delegate]
                        inherit_test_references(child, seen | {entry["name"]})
                        entry["tests"] = sorted(set(entry["tests"]) | set(child["tests"]))
                if entry["delegates"] and entry["tests"]:
                    entry["evidence"] = "component-test-references"

            for entry in entries:
                inherit_test_references(entry)
        else:
            for path in sorted((source / "tile_kernels").rglob("*.py")):
                for node in ast.parse(path.read_text()).body:
                    if not isinstance(node, ast.FunctionDef):
                        continue
                    if not any("tilelang.jit" in ast.unparse(d) for d in node.decorator_list):
                        continue
                    files = references(node.name, tests)
                    entries.append(
                        dict(
                            name=node.name,
                            source=str(path.relative_to(root)),
                            evidence="kernel-test-references" if files else "unmapped",
                            tests=files,
                        )
                    )
            if not entries:
                raise ValueError(f"No TileKernels JIT factories found under {source}")
        libraries[library] = dict(revision=revision(source), entries=entries)
    return libraries


def outcome(case: ET.Element) -> str:
    if case.find("failure") is not None or case.find("error") is not None:
        return "failed"
    skipped = case.find("skipped")
    if skipped is not None:
        return "expected-failure" if skipped.get("type") == "pytest.xfail" else "skipped"
    return "passed"


def case_scope(case: ET.Element) -> str:
    name = case.get("name", "")
    if "rejects_cuda_" in name:
        return "rejection"
    if "lowers_to_riscv" in name:
        return "compile-only"
    if "test_public_" in name:
        return "public-api-numerical"
    return "kernel-or-component-numerical"


def matches(case: ET.Element, library: str, ref: str) -> bool:
    file, _, selector = ref.partition("::")
    classname = case.get("classname", "")
    module = Path(file).stem
    return (classname.endswith(f".{library}.{module}") or classname == f"{library}.{module}") and (
        not selector or case.get("name", "").startswith(selector)
    )


def attach_results(libraries: dict, junit: Path) -> None:
    cases = list(ET.parse(junit).getroot().iter("testcase"))
    for library, inventory in libraries.items():
        own_cases = [c for c in cases if f".{library}." in "." + c.get("classname", "")]
        inventory["junit_outcomes"] = dict(Counter(outcome(c) for c in own_cases))
        inventory["junit_passed_scopes"] = dict(Counter(case_scope(c) for c in own_cases if outcome(c) == "passed"))
        for entry in inventory["entries"]:
            for key in ("tests", "public_api_tests"):
                selected = [c for c in own_cases if any(matches(c, library, r) for r in entry.get(key, []))]
                entry[key + "_outcomes"] = dict(Counter(outcome(c) for c in selected))
                entry[key + "_passed_scopes"] = dict(Counter(case_scope(c) for c in selected if outcome(c) == "passed"))


def validate(libraries: dict, test_root: Path = TEST_ROOT) -> None:
    for library, inventory in libraries.items():
        for entry in inventory["entries"]:
            for ref in entry["tests"] + entry.get("public_api_tests", []):
                file, _, selector = ref.partition("::")
                path = test_root / library / file
                if not path.is_file():
                    raise ValueError(f"{entry['name']}: missing test file {path}")
                if selector and ("def " + selector.split("[")[0] + "(") not in path.read_text():
                    raise ValueError(f"{entry['name']}: missing test function {ref}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="Validate schema and references; unmapped entries remain explicit.")
    parser.add_argument(
        "--require-tileops-numerical",
        action="store_true",
        help="Require JUnit numerical evidence for every entry, with no known or observed gaps.",
    )
    args = parser.parse_args()
    libraries = inventories()
    if args.check:
        validate(libraries)
    if args.junit:
        attach_results(libraries, args.junit)
    report = dict(
        scope="References and selected JUnit outcomes only; mixed test files can include other operators. Counts overlap across entries. No certification of all shapes, dtypes, variants, public APIs or performance.",
        libraries=libraries,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for library, inventory in libraries.items():
        print(f"{library}: {len(inventory['entries'])} declarations; {dict(Counter(e['evidence'] for e in inventory['entries']))}")
        if "junit_outcomes" in inventory:
            print(f"  JUnit: {inventory['junit_outcomes']}; passed scopes: {inventory['junit_passed_scopes']}")
    if args.require_tileops_numerical:
        gaps = [
            e["name"]
            for e in libraries["TileOps"]["entries"]
            if e["evidence"] != "kernel-test-references"
            or e["known_gaps"]
            or not e.get("tests_passed_scopes", {}).get("kernel-or-component-numerical")
            or any(e.get("tests_outcomes", {}).get(k) for k in ("failed", "expected-failure", "skipped"))
        ]
        if gaps:
            raise SystemExit("Numerical acceptance incomplete: " + ", ".join(gaps))


if __name__ == "__main__":
    main()
