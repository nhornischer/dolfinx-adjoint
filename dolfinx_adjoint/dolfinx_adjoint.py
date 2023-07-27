from dolfinx_adjoint import graph
import numpy as np

import time

import ctypes
import networkx as nx

def compute_gradient(J, m):
    _graph = graph.get_graph()

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

    # Check if m is in the graph
    if not adjoint_graph.has_node(id(m)):
        raise Exception("There is not adjoint path from the objective function to the variable")

    graph.visualise(adjoint_graph, filename = "adjoint_graph_transformed.pdf")

    tic = time.perf_counter()
    # Possible to do this in parallel
    for edge in adjoint_graph.edges:
        # print("d",adjoint_graph.nodes[edge[1]]["name"],"/d", adjoint_graph.nodes[edge[0]]["name"])
        adjoint = graph.get_edge_attribute(edge[1], edge[0], "adjoint")
        adjoint.calculate_adjoint()

    adjoint_time = time.perf_counter() - tic
    print(f"Adjoint time: {adjoint_time:0.4f} seconds")

    tic = time.perf_counter()
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
    
    gradient_time = time.perf_counter() - tic
    print(f"Gradient time: {gradient_time:0.4f} seconds")
    
    return gradient