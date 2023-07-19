from dolfinx_adjoint import graph
import numpy as np

import ctypes
import networkx as nx

def compute_gradient(J, m):
    _graph = graph.get_graph()
    functional_node = _graph.nodes[id(J)]
    variable_node = _graph.nodes[id(m)]
    functional_node["name"] =  functional_node["name"] +"(J)"
    variable_node["name"] = variable_node["name"] + " (m)"


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


    # Possible to do this in parallel
    for node_id in adjoint_graph.nodes:
        # Get predeccessor
        predeccessors = list(adjoint_graph.predecessors(node_id))

        adjoint = graph.get_attribute(node_id, "adjoint")
        if len(predeccessors) != 0:
            for predeccessor in predeccessors:
                print(f"Calculating: d{_graph.nodes[node_id]['name']}/d{_graph.nodes[predeccessor]['name']}")
                adjoint.calculate_adjoint(predeccessor)

    # Calculate the gradient
    def _calculate_gradient(node_id):
        adjoint = graph.get_attribute(node_id, "adjoint")
        predeccessors = list(adjoint_graph.predecessors(node_id))
        if len(predeccessors) == 0:
            return 1.0
        else:
            deeper = _calculate_gradient(predeccessors[0])
            current = adjoint.get_adjoint_value(predeccessors[0])
            if type(current) == float or type(deeper) == float:
                return current * deeper
            else:
                return deeper @ current

    gradient = _calculate_gradient(id(J))   
    
    return gradient