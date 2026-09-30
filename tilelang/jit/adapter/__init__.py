from tvm.target import Target

from .base import BaseKernelAdapter, CachedTextSource  # noqa: F401
from .tvm_ffi import TVMFFIKernelAdapter  # noqa: F401
from .cython import CythonKernelAdapter  # noqa: F401
from .nvrtc import NVRTCKernelAdapter  # noqa: F401
from .torch import MetalKernelAdapter  # noqa: F401
from .cutedsl import CuTeDSLKernelAdapter  # noqa: F401


def get_cython_adapter_class(target: Target) -> type[CythonKernelAdapter]:
    """Select the source compiler and launcher for a normalized target."""
    from .utils import is_pto_target

    if is_pto_target(target):
        from .pto.adapter import PTOCythonKernelAdapter

        return PTOCythonKernelAdapter
    return CythonKernelAdapter


def get_tvm_ffi_adapter_class(target: Target) -> type[TVMFFIKernelAdapter]:
    """Select target-specific tensor and stream interop without probing devices."""
    from .utils import is_pto_target

    if is_pto_target(target):
        from .pto.tvm_ffi import PTOTVMFFIKernelAdapter

        return PTOTVMFFIKernelAdapter
    return TVMFFIKernelAdapter
