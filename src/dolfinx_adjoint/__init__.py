import sys

__version__ = "0.3.0"
__author__ = "Niklas Hornischer"
__email__ = "nh605@cam.ac.uk"

from . import fem, nls
from .edge import Edge
from .graph import Graph
from .node import AbstractNode, Node
