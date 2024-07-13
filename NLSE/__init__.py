"""
NLSE.

A package for solving the Nonlinear Schrödinger Equation (NLSE) using the
Split-Step Fourier method.
"""

__version__ = "2.2.0"
__author__ = "Tangui Aladjidi"
__license__ = "MIT"
__credits__ = "Laboratoire Kastler Brossel, Paris, France"
__email__ = "tangui.aladjidi@lkb.upmc.fr"


from . import utils
from .nlse import NLSE
from .nlse_1d import NLSE_1d
from .nlse_3d import NLSE_3d
from .cnlse import CNLSE
from .cnlse_1d import CNLSE_1d
from .gpe import GPE
from .ddgpe import DDGPE
from .callbacks import *
