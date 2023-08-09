"""
This class provides the algorithms and data structures for the computational graph.
The graph is constructed using the networkx library and is a directed graph.
"""

import networkx as nx
from .node import Node, AbstractNode
from .edge import Edge

class Graph:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def add_node(self, node : Node):
        self.nodes.append(node)
    
    def add_edge(self, edge : Edge):
        self.edges.append(edge)

    def get_node(self, id : int):
        for node in self.nodes:
            if node.id == id:
                return node
        return None
    
    def get_edge(self, predecessor : int, successor : int):
        for edge in self.edges:
            if edge.predecessor.id == predecessor and edge.successor.id == successor:
                return edge
        return None
    
    def to_networkx(self):
        nx_graph = nx.DiGraph()
        for node in self.nodes:
            nx_graph.add_node(node.id, name = node.name)
            if type(node) == AbstractNode:
                nx_graph.nodes[node.id]["color"] = 'pink'
            else:
                nx_graph.nodes[node.id]["color"] = 'lightblue'

        for edge in self.edges:
            if not edge.__class__.__name__ == "Edge":
                tag = edge.__class__.__name__
            else:
                tag = ""
            if hasattr(edge, "marked"):
                color = 'black'
            else:
                color = 'grey'
            nx_graph.add_edge(edge.predecessor.id, edge.successor.id, tag = tag, color = color)
        return nx_graph
    
    def visualise(self, filename = "graph.pdf"):
        """Visualise the graph"""
        import matplotlib.pyplot as plt
        # plt.figure(figsize=(10,8))
        plt.figure()
        nx_graph = self.to_networkx()
        labels = nx.get_node_attributes(nx_graph, "name")
        edge_labels = nx.get_edge_attributes(nx_graph, "tag")
        edge_colors = nx.get_edge_attributes(nx_graph, "color")
        node_colors = nx.get_node_attributes(nx_graph, "color")
        nx.draw_shell(nx_graph, labels=labels,node_color = node_colors.values(),  edge_color = edge_colors.values(), with_labels=True)
        nx.draw_networkx_edge_labels(nx_graph, pos=nx.shell_layout(nx_graph), edge_labels=edge_labels)
        plt.savefig(filename)
    
    def backprop(self, function_id, variable_id):
        self.reset_grads()
        function_node = self.get_node(function_id)
        grad_func = function_node.get_gradFuncs()[0]
        grad_func(1.0)
        return self.get_node(variable_id).get_grad()

    def get_backpropagation_path(self, from_id, to_id):
        nx_graph = self.to_networkx()
        paths = nx.all_simple_paths(nx_graph, from_id, to_id)
        edge_paths = []
        for path in paths:
            edge_path = []
            for i in range(1, len(path)):
                edge = self.get_edge(path[-(i+1)],path[-i])
                edge.marked = True
                edge_path.append(edge)
            edge_paths.append(edge_path)
        return edge_paths
    
    def reset_grads(self):
        for node in self.nodes:
            try:
                node.reset_grad()
            except:
                pass