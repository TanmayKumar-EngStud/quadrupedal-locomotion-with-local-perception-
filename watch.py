"""Watch the trained controller on one training terrain; handles macOS's mjpython requirement.

    python watch.py --terrain stairs_up --level 5      (see --help for all options)
"""
from pathlib import Path
import os
import runpy
import sys
import sysconfig

if __name__ == "__main__":
    args = ["-m", "quadruped_nav.watch", *sys.argv[1:]]
    if sys.platform == "darwin" and "--headless" not in sys.argv and "--help" not in sys.argv and "-h" not in sys.argv:
        # mjpython sees the venv symlink rather than uv's real interpreter path.
        libdir = sysconfig.get_config_var("LIBDIR")
        if libdir:
            existing = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "/usr/local/lib:/usr/lib")
            os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = f"{libdir}:{existing}"
        launcher = Path(sys.executable).parent / "mjpython"
        sys.argv = [str(launcher), *args]
        runpy.run_path(str(launcher), run_name="__main__")
    else:
        sys.argv = ["quadruped_nav.watch", *sys.argv[1:]]
        runpy.run_module("quadruped_nav.watch", run_name="__main__")
