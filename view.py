"""Portable viewer launcher, including uv's macOS shared-library lookup fix."""
from pathlib import Path
import os
import runpy
import sys
import sysconfig

if __name__ == "__main__":
    args = ["-m", "quadruped_nav", "view", *sys.argv[1:]]
    if sys.platform == "darwin":
        # mjpython sees the venv symlink rather than uv's real interpreter path.
        # Resolve the real libpython directory before its native trampoline runs.
        libdir = sysconfig.get_config_var("LIBDIR")
        if libdir:
            existing = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "/usr/local/lib:/usr/lib")
            os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = f"{libdir}:{existing}"
        launcher = Path(sys.executable).parent / "mjpython"
        sys.argv = [str(launcher), *args]
        runpy.run_path(str(launcher), run_name="__main__")
    else:
        sys.argv = ["quadruped_nav", "view", *sys.argv[1:]]
        runpy.run_module("quadruped_nav", run_name="__main__")
