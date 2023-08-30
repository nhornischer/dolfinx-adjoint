"""
This class provides the algorithms and data structures for the computational graph.
The graph is constructed using the networkx library and is a directed graph.
"""

import networkx as nx
from .node import Node, AbstractNode
from .edge import Edge

import ctypes

class Graph:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def add_node(self, node : Node):
        self.nodes.append(node)
    
    def add_edge(self, edge : Edge):
        self.edges.append(edge)

    def get_node(self, id : int, version = None):
        # Get all nodes with the given id
        node_versions = {}
        for node in self.nodes:
            if node.id == id:
                node_versions[node.version] = node
        if node_versions == {}: 
            return None
        # If no version is given, return the latest version
        if version is None:
            latest_version = max(node_versions.keys())
            return node_versions[latest_version]
        # If a version is given, return the node with the given version
        elif version in node_versions.keys():
            return node_versions[version]
        else: 
            return None
    def remove_edge(self, edge : Edge):
        self.edges.remove(edge)
        for node in self.nodes:
            if edge in node.get_gradFuncs():
                node.remove_gradFunc(edge)
        for edge in self.edges:
            if edge in edge.next_functions:
                edge.remove_next_function(edge)

    def get_edge(self, predecessor : Node, successor : Node):
        for edge in self.edges:
            if edge.predecessor == predecessor and edge.successor == successor:
                return edge
        return None
    
    def print(self, detailed = False):
        print("#"*64)
        print(f"Graph object with {len(self.nodes)} nodes and {len(self.edges)} edges.")
        print("Nodes:")
        for node in self.nodes:
            print(f"\t{node}")
        print("Edges:")
        for edge in self.edges:
            print(f"\t{edge}")
        if not detailed:
            return
        print("Gradient functions:")
        for node in self.nodes:
            if node.get_gradFuncs() == []:
                continue
            print(f"\t{node}")
            for gradFunc in node.get_gradFuncs():
                print(f"\t\t{gradFunc}")
        print("Next functions:")
        for edge in self.edges:
            if edge.next_functions == []:
                continue
            print(f"\t{edge}")
            for next_function in edge.next_functions:
                print(f"\t\t{next_function}")
        print("#"*64)

    def __str__(self):
        return f"Graph object with {len(self.nodes)} nodes and {len(self.edges)} edges."
    
    def to_networkx(self):
        nx_graph = nx.DiGraph()
        for node in self.nodes:
            nx_graph.add_node(id(node), name = node.name, node = node)
            if type(node) == AbstractNode:
                nx_graph.nodes[id(node)]["color"] = 'pink'
            else:
                nx_graph.nodes[id(node)]["color"] = 'lightblue'

        for edge in self.edges:
            if not edge.__class__.__name__ == "Edge":
                tag = edge.__class__.__name__
            else:
                tag = ""
            if hasattr(edge, "marked"):
                color = 'black'
            else:
                color = 'grey'
            nx_graph.add_edge(id(edge.predecessor), id(edge.successor), tag = tag, color = color, edge = edge)
        return nx_graph
    
    def visualise(self, filename = "graph.pdf", style = "planar"):
        """Visualise the graph"""
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,8))
        nx_graph = self.to_networkx()
        labels = nx.get_node_attributes(nx_graph, "name")
        edge_labels = nx.get_edge_attributes(nx_graph, "tag")
        edge_colors = nx.get_edge_attributes(nx_graph, "color")
        node_colors = nx.get_node_attributes(nx_graph, "color")
        if style == "planar":
            nx.draw_planar(nx_graph, labels=labels, node_color = node_colors.values(),  edge_color = edge_colors.values(), with_labels=True)
            edge_pos = nx.planar_layout(nx_graph)
        elif style == "shell":
            nx.draw_shell(nx_graph, labels=labels, node_color = node_colors.values(),  edge_color = edge_colors.values(), with_labels=True)
            edge_pos = nx.shell_layout(nx_graph)
        else:
            print("Given style is not implemented. Using spring layout")
            nx.draw(nx_graph, labels=labels, node_color = node_colors.values(),  edge_color = edge_colors.values(), with_labels=True)
            edge_pos = nx.spring_layout(nx_graph)
        nx.draw_networkx_edge_labels(nx_graph, pos=edge_pos, edge_labels=edge_labels)
        plt.savefig(filename)
    
    def backprop(self, function_id, variable_id = None):
        self.reset_grads()
        function_node = self.get_node(function_id)
        if variable_id is not None:
            variable_node = self.get_node(variable_id)
            self.get_backpropagation_path(id(variable_node), id(function_node))
        else:
            for edge in self.edges:
                edge.marked = True
        grad_func = function_node.get_gradFuncs()[0]
        grad_func(1.0)
        return self.get_node(variable_id).get_grad()

    def get_backpropagation_path(self, from_id, to_id):
        for edge in self.edges:
            edge.marked = False
        nx_graph = self.to_networkx()
        paths = nx.all_simple_paths(nx_graph, from_id, to_id)
        # Create backpropagation graph based on the paths
        for path in paths:
            for i in range(len(path)-1):
                edge = nx_graph[path[i]][path[i+1]]["edge"]
                edge.marked = True
    
    def reset_grads(self):
        for node in self.nodes:
            try:
                node.reset_grad()
            except:
                pass

    def clear(self):
        for edge in self.edges:
            del edge
        for node in self.nodes:
            del node
        self.edges = []
        self.nodes = []