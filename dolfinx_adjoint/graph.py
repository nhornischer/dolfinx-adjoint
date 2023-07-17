"""
This class provides the algorithms and data structures for the computational graph.
The graph is constructed using the networkx library and is a directed graph.
"""

import networkx as nx

# The main graph object, this object is used to store and manipulate the graph
_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = nx.DiGraph()
    return _graph

def add_node(node):
    """Add a node to the graph"""
    graph = get_graph()
    graph.add_node(node)

def add_edge(node1, node2):
    """Add an edge to the graph"""
    graph = get_graph()
    graph.add_edge(node1, node2)

def visualise(graph = None, filename = "graph.png"):
    """Visualise the graph"""
    import matplotlib.pyplot as plt
    plt.figure()
    if graph is None:
        graph = get_graph()
    labels = nx.get_node_attributes(graph, "name")
    col = nx.get_node_attributes(graph, "color")
    nx.draw(graph, labels=labels)
    plt.savefig(filename)

class Node:

    def __init__(self, name):

        self.name = name


    