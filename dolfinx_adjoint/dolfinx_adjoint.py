from dolfinx_adjoint import graph
import numpy as np

import time

import ctypes
import networkx as nx

def compute_gradient(J, m):
    tic = time.perf_counter()
    _graph = graph.get_graph()

    adjoint_path = []
    for path in nx.all_simple_paths(_graph, source=id(m), target=id(J)):
        adjoint_path += path
    adjoint_graph = _graph.subgraph(adjoint_path).copy()
    graph.visualise(adjoint_graph, filename = "adjoint_graph.pdf")

    circles = list(nx.simple_cycles(adjoint_graph))
    # print(list(circles))
    # Obtain the nodes on the circle that are connected to other node outside the circle
    start_nodes = []
    end_nodes = []
    for circle in circles:
        for node in circle:
            for adjoint_node in adjoint_graph.predecessors(node):
                if not adjoint_node in circle:
                    start_nodes.append(node)
            for adjoint_node in adjoint_graph.successors(node):
                if not adjoint_node in circle:
                    end_nodes.append(node)

    # Remove the nodes of the circle that are not a start or end node
    for circle in circles:
        for node in circle:
            if not node in start_nodes and not node in end_nodes:
                adjoint_graph.remove_node(node)
    for i, node in enumerate(start_nodes):
        adjoint_graph.add_edge(node, end_nodes[i], name="solving")
        adjoint_graph.remove_edge(end_nodes[i], node)

    graph.visualise(adjoint_graph, filename = "adjoint_graph_transformed.pdf")

    graph_time = time.perf_counter() - tic
    print(f"Graph time: {graph_time:0.4f} seconds")

    tic = time.perf_counter()
    # Possible to do this in parallel
    for edge in adjoint_graph.edges:
        print("d",adjoint_graph.nodes[edge[1]]["name"],"/d", adjoint_graph.nodes[edge[0]]["name"])
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