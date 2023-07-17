from dolfinx_adjoint.graph import get_graph, visualise

import ctypes
import networkx as nx

def compute_gradient(J, m):
    graph = get_graph()
    functional_node = graph.nodes[id(J)]
    variable_node = graph.nodes[id(m)]
    functional_node["name"] =  functional_node["name"] +"(J)"
    variable_node["name"] = variable_node["name"] + " (m)"

    adjoint_path = nx.shortest_path(graph, source=id(m), target=id(J))
    adjoint_graph = graph.subgraph(adjoint_path).copy()
    visualise(adjoint_graph, filename = "adjoint_graph")

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

    variable_object = ctypes.cast(id(m), ctypes.py_object).value

    visualise(adjoint_graph, filename = "adjoint_graph_transformed")


    # Possible to do this in parallel
    for node in adjoint_graph.nodes:
        # Get predeccessor
        adjoint_node = list(adjoint_graph.predecessors(node))
        node = adjoint_graph.nodes[node]
        if len(adjoint_node) != 0:

            print(f"Calculating d{node['name']}/d{adjoint_graph.nodes[adjoint_node[0]]['name']}")

            adjoint_variable = ctypes.cast(adjoint_node[0], ctypes.py_object).value

            if "adjoint" in node:
                adjoint_value = node["adjoint"](adjoint_variable)
                node["adjoint_value"] = adjoint_value
        
    adjoints = list(nx.get_node_attributes(adjoint_graph, "adjoint_value").values())

    # Concatenate all the adjoint values
    gradient = 1.0
    for adjoint in adjoints:
        if type(adjoint)==float or type(gradient)==float:
            gradient = gradient * adjoint
        else:
            gradient = gradient @ adjoint
    
    return gradient