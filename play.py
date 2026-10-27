"""Local policy viewer launcher; handles macOS mjpython requirements."""
from pathlib import Path
import os,runpy,sys,sysconfig
if __name__=='__main__':
    args=['-m','quadruped_nav.evaluate',*sys.argv[1:],'--viewer']
    if sys.platform=='darwin':
        libdir=sysconfig.get_config_var('LIBDIR')
        if libdir:os.environ['DYLD_FALLBACK_LIBRARY_PATH']=f"{libdir}:{os.environ.get('DYLD_FALLBACK_LIBRARY_PATH','/usr/local/lib:/usr/lib')}"
        launcher=Path(sys.executable).parent/'mjpython';sys.argv=[str(launcher),*args];runpy.run_path(str(launcher),run_name='__main__')
    else:
        sys.argv=['quadruped_nav.evaluate',*sys.argv[1:],'--viewer'];runpy.run_module('quadruped_nav.evaluate',run_name='__main__')
