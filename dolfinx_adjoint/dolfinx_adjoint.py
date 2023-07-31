from dolfinx_adjoint import graph
import numpy as np

import ctypes
import networkx as nx

def compute_gradient(J, m):

    print("###########Computing Gradient###########")
    _graph = graph.get_graph()

    assert _graph.has_node(id(J)), "The objective function is not in the graph"
    assert _graph.has_node(id(m)), "The variable is not in the graph. Make sure the variable is created using a overloaded dolfinx method."

    adjoint_path = []
    for path in nx.all_simple_paths(_graph, source=id(m), target=id(J)):
        adjoint_path += path
    adjoint_graph = _graph.subgraph(adjoint_path).copy()
    graph.visualise(adjoint_graph, filename = "adjoint_graph.pdf")

    # Remove all the edges that are not tagged with "explicit or implicit"
    for edge in list(adjoint_graph.edges):
        if adjoint_graph.edges[edge]["tag"] != "explicit" and adjoint_graph.edges[edge]["tag"] != "implicit":
            adjoint_graph.remove_edge(edge[0], edge[1])

    # Remove all the nodes that are not connected to the objective function
    for node in list(adjoint_graph.nodes):
        if not nx.has_path(adjoint_graph, node, id(J)):
            adjoint_graph.remove_node(node)
            
    graph.visualise(adjoint_graph, filename = "adjoint_graph_transformed.pdf")

    # Possible to do this in parallel
    for edge in adjoint_graph.edges:
        # print("d",adjoint_graph.nodes[edge[1]]["name"],"/d", adjoint_graph.nodes[edge[0]]["name"])
        adjoint = graph.get_edge_attribute(edge[1], edge[0], "adjoint")
        adjoint.calculate_adjoint()

    # Calculate the gradient
    def _calculate_gradient(node_id):
        edges = adjoint_graph.in_edges(node_id)
        if len(edges) == 0:
            return 1.0
        else:
            gradient = 0.0
            for edge in edges:
                deeper = _calculate_gradient(edge[0])
                adjoint = graph.get_edge_attribute(node_id, edge[0], "adjoint")
                if type(adjoint.value) == float or type(deeper) == float:
                    gradient += adjoint.value * deeper
                else:
                    gradient += deeper @ adjoint.value
            return gradient


    gradient = _calculate_gradient(id(J))   
    
    return gradient