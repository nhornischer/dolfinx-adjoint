"""
This class provides the algorithms and data structures for the computational graph.
The graph is constructed using the networkx library and is a directed graph.
"""

import networkx as nx
import ctypes

# The main graph object, this object is used to store and manipulate the graph
_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = nx.DiGraph()
    return _graph

def add_node(node : int, **kwargs):
    """Add a node to the graph"""
    graph = get_graph()
    graph.add_node(node, **kwargs)

def add_edge(node1 : int, node2 : int):
    """Add an edge to the graph"""
    graph = get_graph()
    graph.add_edge(node1, node2)

def add_attribute(node : int, key : str, value):
    """Add an attribute to a node"""
    graph = get_graph()
    graph.nodes[node][key] = value

def get_attribute(node : int, key : str):
    """Get an attribute from a node"""
    graph = get_graph()
    return graph.nodes[node][key]
    
def add_incoming_edges(node : int, objects : list):
    for object in objects:
        add_edge(id(object), node)

def visualise(graph = None, filename = "graph.pdf"):
    """Visualise the graph"""
    import matplotlib.pyplot as plt
    plt.figure()
    if graph is None:
        graph = get_graph()
    labels = nx.get_node_attributes(graph, "name")
    col = nx.get_node_attributes(graph, "color")
    nx.draw(graph, labels=labels)
    plt.savefig(filename)

class Adjoint:

    def __init__(self, object):
        self.id = id(object)
        self.adjoints = {}

    def set_adjoint_method(self, function : callable):
        """Set the adjoint method for the object"""
        graph = get_graph()
        self.adjoint_method = function
        # Temporary fix
        graph.nodes[self.id]["adjoint"] = function

    def set_adjoint_value(self, value, adjoint_node : int):
        """Set the adjoint value for the object"""
        self.adjoints[str(adjoint_node)] = value

    def get_adjoint_value(self, adjoint_node : int):
        """Get the adjoint value for the object"""
        return self.adjoints[str(adjoint_node)]

    def calculate_adjoint(self, adjoint_node : int):
        """Calculate the adjoint value for the object"""
        adjoint_object = ctypes.cast(adjoint_node, ctypes.py_object).value
        adjoint_value = self.adjoint_method(adjoint_object)

        self.set_adjoint_value(adjoint_value, adjoint_node)

    def add_to_graph(self):
        """Add the object to the graph"""
        add_attribute(self.id, "adjoint", self)


    