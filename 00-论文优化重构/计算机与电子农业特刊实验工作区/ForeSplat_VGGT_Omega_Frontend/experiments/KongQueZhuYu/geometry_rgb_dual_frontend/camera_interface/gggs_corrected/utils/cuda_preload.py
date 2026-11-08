"""Preload the shared libraries the GGGS CUDA extensions link against.

``diff_gaussian_rasterization._C`` and ``simple_knn._C`` link against
``libc10.so`` (inside the torch package) and, through ``libcusparse``, against
``libnvJitLink.so.12`` (pip ``nvidia-nvjitlink``).  Neither directory is on the
default loader path, and editing ``LD_LIBRARY_PATH`` at runtime does not affect
``dlopen``, so the libraries are loaded explicitly with ``RTLD_GLOBAL`` before
the extensions are imported.
"""

from __future__ import annotations

import ctypes
import glob
import os


def preload_cuda_ext_deps() -> None:
    import torch  # noqa: F401 - torch must be imported first

    torch_dir = os.path.dirname(torch.__file__)
    site_dir = os.path.dirname(os.path.dirname(torch_dir))
    for pattern in (
        os.path.join(site_dir, "nvidia", "nvjitlink", "lib", "libnvJitLink.so*"),
        os.path.join(torch_dir, "lib", "libc10.so"),
    ):
        for path in sorted(glob.glob(pattern)):
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
