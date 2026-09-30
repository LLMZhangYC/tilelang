"""Backend adapter dispatch and artifact handoff without device execution."""

from types import SimpleNamespace

import pytest

from tilelang import tvm
from tilelang.jit.adapter import get_cython_adapter_class, get_tvm_ffi_adapter_class
from tilelang.jit.adapter.ascend.adapter import AscendCythonKernelAdapter
from tilelang.jit.adapter.ascend.libgen import AscendLibraryGenerator
from tilelang.jit.adapter.ascend import tvm_ffi as ascend_ffi
from tilelang.jit.adapter.ascend.tvm_ffi import AscendTVMFFIKernelAdapter
from tilelang.jit.adapter.cython import adapter as cython_adapter
from tilelang.jit.adapter.cython import CythonKernelAdapter
from tilelang.jit.adapter.pto.adapter import PTOCythonKernelAdapter
from tilelang.jit.adapter.pto.libgen import PTOLibraryGenerator
from tilelang.jit.adapter.pto.tvm_ffi import PTOTVMFFIKernelAdapter
from tilelang.jit.adapter import tvm_ffi
from tilelang.jit.adapter.tvm_ffi import TVMFFIKernelAdapter


@pytest.mark.parametrize(
    "kind,keys,cython_cls,ffi_cls",
    [
        ("cuda", (), CythonKernelAdapter, TVMFFIKernelAdapter),
        ("hip", (), CythonKernelAdapter, TVMFFIKernelAdapter),
        ("c", (), CythonKernelAdapter, TVMFFIKernelAdapter),
        ("ascend", (), AscendCythonKernelAdapter, AscendTVMFFIKernelAdapter),
        ("ascend", ("pto",), PTOCythonKernelAdapter, PTOTVMFFIKernelAdapter),
    ],
)
def test_adapter_selection(kind, keys, cython_cls, ffi_cls):
    target = SimpleNamespace(kind=SimpleNamespace(name=kind), keys=keys)
    assert get_cython_adapter_class(target) is cython_cls
    assert get_tvm_ffi_adapter_class(target) is ffi_cls


def test_pto_compilation_receives_device_and_host_sources(monkeypatch):
    target = SimpleNamespace(kind=SimpleNamespace(name="ascend"), keys=("pto",))
    adapter = object.__new__(PTOCythonKernelAdapter)
    adapter.lib_generator = PTOLibraryGenerator(target)
    source = "def first(a): pass\ndef second(a): pass"
    names = ["second", "first"]
    adapter.wrapper = SimpleNamespace(source_wrapper=SimpleNamespace(pto_kernel_source=source, pto_kernel_names=names))
    adapter.host_kernel_source = "host launcher"
    compiled = []

    def compile_lib():
        generator = adapter.lib_generator
        compiled.append((generator.lib_code, generator.pto_kernel_source, generator.pto_kernel_names))

    monkeypatch.setattr(adapter.lib_generator, "compile_lib", compile_lib)
    adapter._compile_library()
    names.clear()

    assert compiled == [("host launcher", source, ["second", "first"])]


@pytest.mark.parametrize("adapter_cls", [AscendCythonKernelAdapter, PTOCythonKernelAdapter])
def test_cython_cache_load_uses_backend_generator(monkeypatch, adapter_cls):
    loaded = []
    generator_cls = adapter_cls.library_generator_class
    library = SimpleNamespace(init=lambda: 0, get_last_error=SimpleNamespace())

    def load_lib(self, lib_path=None):
        loaded.append((self, lib_path))
        return library

    def unexpected_compile(*args, **kwargs):
        pytest.fail("Loading a cached shared library must not invoke a compiler")

    monkeypatch.setattr(generator_cls, "load_lib", load_lib)
    monkeypatch.setattr(generator_cls, "compile_lib", unexpected_compile)
    for name in (
        "_process_dynamic_symbolic",
        "_process_dynamic_symbolic_sources",
        "_process_scalar_param_vars",
        "_process_buffer_dtype",
        "_process_param_storage_metadata",
        "_process_ptr_map",
        "_process_buffer_device",
    ):
        monkeypatch.setattr(adapter_cls, name, lambda self: {})
    monkeypatch.setattr(adapter_cls, "_process_static_buffer_infos", lambda self: ({}, {}, [], {}))
    monkeypatch.setattr(adapter_cls, "_post_init", lambda self: None)

    class CachedWrapper:
        def __init__(self, *args):
            pass

        def __getattr__(self, name):
            assert name.startswith("set_")
            return lambda value: None

    monkeypatch.setattr(cython_adapter, "CythonKernelWrapper", CachedWrapper)
    configs = {"test_config": True}
    flags = ["-O2"]
    adapter = adapter_cls.from_database(
        params=[],
        result_idx=[],
        # The storage and IR analysis is mocked, so this works on CUDA-only builds.
        target="c",
        func_or_mod=tvm.IRModule(),
        host_kernel_source="cached host",
        device_kernel_source="cached device",
        kernel_lib_path="cached_kernel.so",
        pass_configs=configs,
        compile_flags=flags,
    )

    assert isinstance(adapter, adapter_cls)
    assert isinstance(adapter.lib_generator, generator_cls)
    assert loaded == [(adapter.lib_generator, "cached_kernel.so")]
    assert adapter.lib_generator.pass_configs == configs
    assert adapter.lib_generator.compile_flags == flags
    assert adapter.lib is library


@pytest.mark.parametrize("cached", [False, True])
def test_npu_stream_exchange_is_lazy_and_installed_once(monkeypatch, cached):
    calls = []
    monkeypatch.setattr(ascend_ffi, "_install_torch_stream_exchange", lambda: calls.append("install"))
    if cached:
        executable = object()
        monkeypatch.setattr(tvm_ffi.runtime, "load_module", lambda path: executable)
        monkeypatch.setattr(AscendTVMFFIKernelAdapter, "_uses_ffi_callee_allocated_output_abi", lambda self: False)
        monkeypatch.setattr(AscendTVMFFIKernelAdapter, "_process_dynamic_symbolic", lambda self: {})
        monkeypatch.setattr(AscendTVMFFIKernelAdapter, "_post_init", lambda self: None)
        adapter = AscendTVMFFIKernelAdapter.from_database(
            params=[],
            result_idx=[],
            target="c",
            func_or_mod=tvm.IRModule(),
            host_kernel_source="cached host",
            device_kernel_source="cached device",
            kernel_lib_path="cached_kernel.so",
        )
        assert adapter.executable is executable
    else:
        adapter = object.__new__(AscendTVMFFIKernelAdapter)

    assert calls == []
    adapter._prepare_torch_device(SimpleNamespace(type="cpu"))
    assert calls == []
    adapter._prepare_torch_device(SimpleNamespace(type="npu"))
    adapter._prepare_torch_device(SimpleNamespace(type="npu"))
    assert calls == ["install"]


@pytest.mark.parametrize("names", [[], ["kernel", "kernel"], [""]])
def test_pto_rejects_invalid_entry_points(names):
    generator = PTOLibraryGenerator(SimpleNamespace())
    with pytest.raises(RuntimeError):
        generator.update_pto_kernels("def kernel(): pass", names)


@pytest.mark.parametrize("keep_files", [False, True])
def test_pto_compile_failure_honors_temp_file_policy(monkeypatch, tmp_path, keep_files):
    generator = PTOLibraryGenerator(SimpleNamespace())
    removed = []
    temp_dir = SimpleNamespace(temp_dir=str(tmp_path), remove=lambda: removed.append(True))
    generator._pto_temp_dir = temp_dir
    generator._keep_pto_temp_files = keep_files
    generator.srcpath = str(tmp_path / "launch.cpp")
    generator.libpath = str(tmp_path / "lib_kernel.so")

    def fail_compile():
        raise RuntimeError("compiler failed")

    monkeypatch.setattr(generator, "_compile_pto_lib", fail_compile)
    with pytest.raises(RuntimeError, match="compiler failed"):
        generator.compile_lib()

    assert removed == ([] if keep_files else [True])
    assert (generator._pto_temp_dir is temp_dir) == keep_files
    assert (generator.srcpath is not None) == keep_files
    assert (generator.libpath is not None) == keep_files


def test_pto_ffi_uses_storage_shape_for_its_target():
    target = SimpleNamespace(kind=SimpleNamespace(name="ascend"), keys=("pto",))
    adapter = object.__new__(PTOTVMFFIKernelAdapter)
    adapter.target = target
    seen = []

    def storage_shape(*, target):
        seen.append(target)
        return [16, 32]

    adapter.params = [SimpleNamespace(storage_shape=storage_shape)]
    assert adapter._get_param_shapes() == [[16, 32]]
    assert seen == [target]


def test_ascend_compile_keeps_repeated_flags_and_timeout(monkeypatch):
    from tilelang.contrib import bisheng

    target = SimpleNamespace(kind=SimpleNamespace(name="ascend"), keys=())
    generator = AscendLibraryGenerator(target)
    generator.assign_compile_flags(["-mllvm", "first=true", "-mllvm", "second=true"])
    monkeypatch.setattr(bisheng, "find_bisheng_path", lambda: "bisheng")
    monkeypatch.setattr(bisheng, "get_target_npu_arch", lambda target: "dav-test")
    monkeypatch.setattr(bisheng, "get_bisheng_compile_options", lambda arch: [f"--npu-arch={arch}"])
    captured = []

    def run_compile(command, src, libpath, timeout):
        captured.append((command, libpath, timeout))
        src.close()
        # The fake compiler does not consume this file.
        from pathlib import Path

        Path(src.name).unlink()

    monkeypatch.setattr(generator, "_run_compile", run_compile)
    generator.compile_lib(timeout=12)
    command, libpath, timeout = captured[0]
    assert command[:6] == ["bisheng", "--npu-arch=dav-test", "-mllvm", "first=true", "-mllvm", "second=true"]
    assert command[-2:] == ["-o", libpath]
    assert timeout == 12
