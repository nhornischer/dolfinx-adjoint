import sys


dolfinx_modules = []
# Remove all modules from the modules list except dolfinx modules
for i,module in enumerate(list(sys.modules.keys())):
    if module.startswith("dolfinx"):
        dolfinx_modules.append(module)

# print(dolfinx_modules)

from .dolfinx_adjoint import *
from .graph import *
from . import fem
from . import nls