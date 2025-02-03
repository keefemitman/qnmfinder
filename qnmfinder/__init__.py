import os
from matplotlib import style

from . import model
from . import ringdown
from . import plotting
from . import utils
from . import varpro

from importlib.metadata import version

__version__ = version("qnmfinder")

module_dir = os.path.dirname(__file__)
style.use(os.path.join(module_dir, "style.mplstyle"))
