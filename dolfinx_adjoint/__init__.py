import sys

__version__ = "0.2.0"
__author__ = "Niklas Hornischer"
__email__ = "nh605@cam.ac.uk"

from .node import AbstractNode, Node
from .edge import Edge
from .graph import Graph

from . import fem, nls
from .active_subspace import ASFEniCSx as AS
