from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
import ast
import importlib
import importlib.util
import inspect
import os
import sys
import types
from pathlib import Path
from typing import Any

import torch
import tilelang

REPO_ROOT = Path(__file__).resolve().parents[4]
TILEOPS_ROOT = Path(os.environ.get("TILEOPS_ROOT", REPO_ROOT / "3rdparty" / "TileOPs"))
TILEOPS_SOURCE_ROOT = TILEOPS_ROOT / "src" if (TILEOPS_ROOT / "src" / "tileops").is_dir() else TILEOPS_ROOT
TILEOPS_PACKAGE_ROOT = TILEOPS_SOURCE_ROOT / "tileops"
TILEOPS_KERNEL_ROOT = TILEOPS_PACKAGE_ROOT / "kernels"

source_root = str(TILEOPS_SOURCE_ROOT)
if source_root not in sys.path:
    sys.path.insert(0, source_root)


def _ensure_tvm_tirx_shim() -> None:
    """Expose this TVM revision's ``tir`` module under TileOps' newer name."""
    import tvm
    import tvm.tir

    sys.modules.setdefault("tvm.tirx", tvm.tir)
    sys.modules.setdefault("tvm.tirx.op", importlib.import_module("tvm.tir.op"))
    sys.modules.setdefault("tvm.tirx.stmt_functor", importlib.import_module("tvm.tir.stmt_functor"))
    if not hasattr(tvm, "tirx"):
        tvm.tirx = tvm.tir


def _ensure_torch_library_custom_op_shim() -> None:
    if hasattr(torch.library, "custom_op"):
        return

    def custom_op(*_args, **_kwargs):
        def decorator(fn):
            def register_fake(_fake_fn=None, **_fake_kwargs):
                def fake_decorator(fake_fn):
                    return fake_fn

                if _fake_fn is not None:
                    return _fake_fn
                return fake_decorator

            fn.register_fake = register_fake
            return fn

        return decorator

    torch.library.custom_op = custom_op


def _load_module(module_name: str, file_path: Path):
    if not file_path.is_file():
        legacy_prefixes = {
            "tileops.kernels.deltanet.": "tileops.kernels.linear_attention.deltanet.",
            "tileops.kernels.gated_deltanet.": ("tileops.kernels.linear_attention.gated_deltanet."),
            "tileops.kernels.gla.": "tileops.kernels.linear_attention.gla.",
        }
        for old_prefix, new_prefix in legacy_prefixes.items():
            if module_name.startswith(old_prefix):
                module_name = new_prefix + module_name.removeprefix(old_prefix)
                break
        module_name = module_name.replace(".deltanet.deltanet_fwd", ".deltanet.chunk_fwd")
        module_name = module_name.replace(".deltanet.deltanet_bwd", ".deltanet.chunk_bwd")
        module_name = module_name.replace(".gla.gla_fwd", ".gla.chunk_fwd")
        module_name = module_name.replace(".gla.gla_bwd", ".gla.chunk_bwd")
        return importlib.import_module(module_name)
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _ensure_minimal_tileops_kernel_modules():
    _ensure_torch_library_custom_op_shim()
    _ensure_tvm_tirx_shim()
    importlib.import_module("tileops")
    utils = importlib.import_module("tileops.utils")
    utils.get_sm_version = lambda _index=None: 80
    utils.get_sm_count = lambda _index=None: 1
    # Fix the schedule-selection budget for direct kernel tests. No device
    # allocation or CUDA execution is permitted by this harness.
    utils.get_shared_memory_optin = lambda _index=None: 98304
    kernel_base = importlib.import_module("tileops.kernels.kernel_base")
    # ``supported_archs`` describes CUDA SM compatibility. It cannot classify
    # a non-CUDA backend, so RISC-V lowering tests validate the generated TIR
    # directly and let unsupported GPU intrinsics fail in the backend instead.
    kernel_base.Kernel._check_arch = lambda _self: None
    return kernel_base


def _ensure_elementwise_kernel_module():
    kernel_base = _ensure_minimal_tileops_kernel_modules()
    elementwise = importlib.import_module("tileops.kernels.elementwise")

    # TileOps compiles elementwise kernels eagerly for its CUDA-facing forward
    # methods. RISC-V tests need the undecorated JIT factory so the harness can
    # request target="riscv" explicitly.
    elementwise_base = importlib.import_module("tileops.kernels.elementwise._base")
    elementwise_base._StrategyKernel.init_config = kernel_base.Kernel.init_config
    elementwise_base.MultiInputElementwiseKernel.init_config = kernel_base.Kernel.init_config
    return elementwise


def _ensure_kernel_parent_packages(module_name: str) -> None:
    parent = "tileops.kernels"
    parent_path = TILEOPS_KERNEL_ROOT
    for part in module_name.split(".")[:-1]:
        parent = f"{parent}.{part}"
        parent_path = parent_path / part
        package = sys.modules.setdefault(parent, types.ModuleType(parent))
        package.__path__ = [str(parent_path)]


def _ensure_trace_stub():
    if "tileops.trace" in sys.modules:
        return

    class _TraceStub:
        enabled = False

        @staticmethod
        def out_idx(count, traced):
            if traced:
                return list(range(count + 1))
            return list(range(count))

        @staticmethod
        def finalize(func, **_kwargs):
            return func

        @staticmethod
        def run(compiled, args, **_kwargs):
            return compiled(*args)

        @staticmethod
        def group(*_args, **_kwargs):
            return _NullContext()

        @staticmethod
        def range(*_args, **_kwargs):
            return _NullContext()

        @staticmethod
        def dag(*_args, **_kwargs):
            return None

    class _NullContext:
        def __enter__(self):
            return None

        def __exit__(self, *_exc):
            return False

    trace_pkg = types.ModuleType("tileops.trace")
    trace_pkg.trace = _TraceStub()
    sys.modules["tileops.trace"] = trace_pkg


def _ensure_minimal_pool_kernel_module(module_name: str):
    _ensure_minimal_tileops_kernel_modules()
    return importlib.import_module(f"tileops.kernels.pool.{module_name}")


def _ensure_minimal_norm_kernel_module(module_name: str):
    _ensure_minimal_tileops_kernel_modules()
    return importlib.import_module(f"tileops.kernels.norm.{module_name}")


def _ensure_minimal_reduction_kernel_module(module_name: str):
    _ensure_minimal_tileops_kernel_modules()
    return importlib.import_module(f"tileops.kernels.reduction.{module_name}")


def get_elementwise_kernel_class(name: str):
    module = _ensure_elementwise_kernel_module()
    return getattr(module, name)


def get_pool_kernel_class(module_name: str, class_name: str):
    module = _ensure_minimal_pool_kernel_module(module_name)
    return getattr(module, class_name)


def get_norm_kernel_class(module_name: str, class_name: str):
    module = _ensure_minimal_norm_kernel_module(module_name)
    return getattr(module, class_name)


def get_reduction_kernel_class(module_name: str, class_name: str):
    module = _ensure_minimal_reduction_kernel_module(module_name)
    return getattr(module, class_name)


@lru_cache(maxsize=1)
def _kernel_class_modules() -> dict[str, str]:
    modules: dict[str, str] = {}
    for file_path in TILEOPS_KERNEL_ROOT.rglob("*.py"):
        try:
            tree = ast.parse(file_path.read_text())
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        relative = file_path.relative_to(TILEOPS_PACKAGE_ROOT).with_suffix("")
        parts = relative.parts[:-1] if relative.name == "__init__" else relative.parts
        module_name = ".".join(("tileops", *parts))
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                modules.setdefault(node.name, module_name)
    return modules


def get_kernel_class(module_name: str, class_name: str):
    _ensure_minimal_tileops_kernel_modules()
    del module_name  # Class names are stable while modules were reorganized under src/.
    kernels = importlib.import_module("tileops.kernels")
    try:
        return getattr(kernels, class_name)
    except AttributeError:
        resolved_module = _kernel_class_modules().get(class_name)
        if resolved_module is None:
            raise ImportError(f"TileOps no longer defines kernel class {class_name}") from None
        return getattr(importlib.import_module(resolved_module), class_name)


def tileops_kernel_tir(tileops_kernel: Any):
    cfg = tileops_kernel.config
    jit_kernel = tileops_kernel.kernel

    # TileOps configs use the JIT factory's keyword names. Prefer that direct
    # mapping so kernels whose factory has no ``threads`` argument (for
    # example GemmKernel) are not forced through the legacy shape heuristics.
    parameters = getattr(jit_kernel, "arg_names", None)
    if parameters is None:
        parameters = getattr(getattr(jit_kernel, "func", None), "arg_names", None)
    if parameters is None:
        try:
            parameters = inspect.signature(jit_kernel).parameters
        except (TypeError, ValueError):
            parameters = {}
    if parameters and set(cfg).issubset(parameters):
        return jit_kernel.get_tir(**cfg)

    if {"block_l", "block_p", "block_n", "block_s", "num_stages"} <= set(cfg):
        return jit_kernel.get_tir(
            cfg["block_l"],
            cfg["block_p"],
            cfg["block_n"],
            cfg["block_s"],
            cfg["threads"],
            cfg["num_stages"],
        )
    if {"block_n", "block_p", "block_l"} <= set(cfg):
        return jit_kernel.get_tir(
            cfg["block_n"],
            cfg["block_p"],
            cfg["block_l"],
            cfg["threads"],
        )
    if "block_l" in cfg and "block_s" in cfg and "block_n" in cfg:
        return jit_kernel.get_tir(
            cfg["block_l"],
            cfg["block_s"],
            cfg["block_n"],
            cfg["threads"],
        )
    if "block_l" in cfg:
        if "num_stages" in cfg:
            return jit_kernel.get_tir(
                cfg["block_l"],
                cfg["num_stages"],
                cfg["threads"],
            )
        return jit_kernel.get_tir(cfg["block_l"], cfg["threads"])
    if "block_d" in cfg:
        return jit_kernel.get_tir(cfg["block_d"], cfg["threads"], cfg["vectorize"])
    if "block_m" in cfg and "num_stages" in cfg and "threads" not in cfg:
        return jit_kernel.get_tir(cfg["num_stages"], cfg["block_m"])
    if {"block_n", "reduce_threads", "num_stages"} <= set(cfg):
        return jit_kernel.get_tir(
            cfg["block_n"],
            cfg["reduce_threads"],
            cfg["num_stages"],
        )
    if {"RADIX", "BLOCK_SIZE", "SMEM_INPUT_SIZE", "block_m"} <= set(cfg):
        return jit_kernel.get_tir(
            cfg["RADIX"],
            cfg["BLOCK_SIZE"],
            cfg["SMEM_INPUT_SIZE"],
            cfg["block_m"],
        )
    if {"block_i", "threads"} <= set(cfg):
        return jit_kernel.get_tir(cfg["block_i"], cfg["threads"])
    if {"block_m", "block_n", "num_stages", "threads"} <= set(cfg) and "block_k" not in cfg:
        return jit_kernel.get_tir(
            cfg["block_m"],
            cfg["block_n"],
            cfg["num_stages"],
            cfg["threads"],
        )
    if "block_k" in cfg and "num_stages" in cfg:
        args = [
            cfg["block_m"],
            cfg["block_n"],
            cfg["block_k"],
            cfg["num_stages"],
            cfg["threads"],
        ]
        if "enable_rasterization" in cfg:
            args.append(cfg["enable_rasterization"])
        return jit_kernel.get_tir(*args)
    if "block_n" in cfg:
        if "block_m" in cfg:
            return jit_kernel.get_tir(
                cfg["block_m"],
                cfg["block_n"],
                cfg["threads"],
            )
        if "block_p" in cfg:
            return jit_kernel.get_tir(
                cfg["block_p"],
                cfg["block_n"],
                cfg["threads"],
            )
    if "block_x_b" in cfg and "block_C" in cfg:
        return jit_kernel.get_tir(
            cfg["block_x_b"],
            cfg["block_C"],
            cfg["num_stages"],
            cfg["threads"],
        )
    if "block_h" in cfg:
        return jit_kernel.get_tir(cfg["block_h"], cfg["threads"])
    if "bdim" in cfg:
        return jit_kernel.get_tir(cfg["bdim"], cfg["threads"])
    if "bwidth" in cfg:
        return jit_kernel.get_tir(cfg["bwidth"], cfg["threads"])
    if "block_size" in cfg:
        return jit_kernel.get_tir(cfg["block_size"], cfg["threads"])
    if "block_m" in cfg:
        if set(cfg) == {"block_m"}:
            return jit_kernel.get_tir(cfg["block_m"])
        return jit_kernel.get_tir(cfg["block_m"], cfg["threads"])
    if "num_per_thread" in cfg:
        args = [cfg["threads"], cfg["num_per_thread"]]
        if "steps" in cfg:
            args.append(cfg["steps"])
        return jit_kernel.get_tir(*args)
    return jit_kernel.get_tir(cfg["threads"])


def compile_tileops_kernel(
    tileops_kernel: Any,
    *,
    out_idx: list[int] | int | None = None,
):
    jit_kernel = tileops_kernel.kernel
    return tilelang.compile(
        tileops_kernel_tir(tileops_kernel),
        out_idx=jit_kernel.out_idx if out_idx is None else out_idx,
        target="riscv",
    )


def compile_tileops_jit(
    jit_kernel: Any,
    config: dict[str, Any],
    *,
    out_idx: list[int] | int | None = None,
):
    """Compile a TileOps JIT factory when its public wrapper is CUDA-only."""
    result_idx = getattr(jit_kernel, "out_idx", None) if out_idx is None else out_idx
    return tilelang.compile(jit_kernel.get_tir(**config), out_idx=result_idx, target="riscv")


def run_unary_runtime_compare(
    kernel_cls: type,
    x: torch.Tensor,
    reference: Callable[[torch.Tensor], torch.Tensor],
    *,
    kernel_kwargs: dict[str, Any] | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> None:
    kernel_kwargs = dict(kernel_kwargs or {})
    config = None
    if "strategy" in kernel_kwargs:
        config = {"strategy": kernel_kwargs.pop("strategy")}
    tileops_kernel = kernel_cls(N_total=x.numel(), dtype=x.dtype, config=config, **kernel_kwargs)
    kernel = compile_tileops_kernel(tileops_kernel)
    actual = kernel(x.contiguous().reshape(-1))

    expected = reference(x).reshape(-1)
    # Upstream predicate kernels use byte storage; their public wrappers expose bool.
    if expected.dtype == torch.bool and actual.dtype == torch.int8:
        assert torch.all((actual == 0) | (actual == 1))
        actual = actual.bool()
    torch.testing.assert_close(actual.reshape(expected.shape), expected, rtol=rtol, atol=atol)


def run_binary_runtime_compare(
    kernel_cls: type,
    a: torch.Tensor,
    b: torch.Tensor,
    reference: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    kernel_kwargs: dict[str, Any] | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> None:
    kernel_kwargs = dict(kernel_kwargs or {})
    config = None
    if "strategy" in kernel_kwargs:
        config = {"strategy": kernel_kwargs.pop("strategy")}
    a_flat = a.contiguous().reshape(-1)
    b_flat = b.contiguous().reshape(-1)
    assert a_flat.numel() == b_flat.numel()
    tileops_kernel = kernel_cls(
        (a_flat.numel(),),
        (b_flat.numel(),),
        a.dtype,
        config=config,
        **kernel_kwargs,
    )
    kernel = compile_tileops_kernel(tileops_kernel)
    actual = kernel(a_flat, b_flat)

    expected = reference(a, b).reshape(-1)
    torch.testing.assert_close(actual.reshape(expected.shape), expected, rtol=rtol, atol=atol)


def run_fused_gated_runtime_compare(
    kernel_cls: type,
    x: torch.Tensor,
    reference: Callable[[torch.Tensor], torch.Tensor],
    *,
    m: int,
    n: int,
    kernel_kwargs: dict[str, Any] | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> None:
    kernel_kwargs = dict(kernel_kwargs or {})
    config = None
    if "strategy" in kernel_kwargs:
        config = {"strategy": kernel_kwargs.pop("strategy")}
    tileops_kernel = kernel_cls(m, n, x.dtype, config=config, **kernel_kwargs)
    kernel = compile_tileops_kernel(tileops_kernel)
    actual = kernel(x.contiguous())

    expected = reference(x).reshape(-1)
    torch.testing.assert_close(actual.reshape(expected.shape), expected, rtol=rtol, atol=atol)
