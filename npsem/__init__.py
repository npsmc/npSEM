import sys

try:
    import npsem.models.l63f as mdl_l63
except ModuleNotFoundError:
    print("""\
          Do not forget to build fortran modules
          run `python3 setup.py build_ext --inplace`
          """)
    sys.exit(0)

from .methods.data import Data
from .methods.est  import Est
