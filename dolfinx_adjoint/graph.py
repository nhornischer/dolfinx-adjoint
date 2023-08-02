"""
This class provides the algorithms and data structures for the computational graph.
The graph is constructed using the networkx library and is a directed graph.
"""

import networkx as nx

class Node: 
    def __init__(self, object, **kwargs):
        self.id = id(object)
        self.object = object
        if "name" in kwargs:
            self.name = kwargs["name"]
        else:
            self.name = str(self.object.__module__ + "."+ self.object.__class__.__name__)

    def set_adjoint_value(self, value):
        "Remember to set this value for the first node in the path"
        self.adjoint_value = value

    def get_adjoint_value(self):
        return self.adjoint_value
    
class Edge:
    def __init__(self, predecessor : Node, successor : Node):
        self.predecessor = predecessor
        self.successor = successor
    
    def calculate_tlm(self):
        """
        This method calculates the default tangent linear model (TLM) for
        the edge, which corresponds to the derivative
            ∂successor/∂predecessor = 1.0
        
        This operator is stored in the edge and applied to the adjoint value
        stored in the successor node. The adjoint value of the predeccessor
        node is then calculated as follows:
            adjoint(predecessor) += adjoint(successor) * ∂successor/∂predecessor
        """

        self.tlm = 1.0
        self.predecessor.set_adjoint_value(self.successor.get_adjoint_value() * self.tlm)

    def calculate_adjoint(self):
        """
        This method calculates the default adjoint for the edge, which corresponds
        to the derivative
            ∂predecessor/∂successor = 1.0
            
        This operator is stored in the edge and applied to the adjoint value
        stored in the successor node. The adjoint value of the predecessor
        node is then calculated as follows:
            adjoint(predecessor) += adjoint(successor) * ∂predecessor/∂successor
        """
        self.adjoint = 1.0
        self.predecessor.set_adjoint_value(self.successor.get_adjoint_value() * self.adjoint)

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
        plt.figure()
        nx_graph = self.to_networkx()
        labels = nx.get_node_attributes(nx_graph, "name")
        edge_labels = nx.get_edge_attributes(nx_graph, "tag")
        edge_colors = nx.get_edge_attributes(nx_graph, "color")
        nx.draw_shell(nx_graph, labels=labels, edge_color = edge_colors.values(), with_labels=True)
        nx.draw_networkx_edge_labels(nx_graph, pos=nx.shell_layout(nx_graph), edge_labels=edge_labels)
        plt.savefig(filename)

    def compute_tlm(self, function_id, variable_id):
        # Set the adjoint value of the function to 1.0
        self.get_node(function_id).set_adjoint_value(1.0)
        # Get the paths from the function to the variable
        paths = self.get_backpropagation_path(variable_id, function_id)
        for path in paths:
            for edge in path:
                edge.calculate_tlm()
        
        return self.get_node(variable_id).get_adjoint_value()
    
    def compute_adjoint(self, function_id, variable_id):
        # Set the adjoint value of the function to 1.0
        self.get_node(function_id).set_adjoint_value(1.0)
        # Get the paths from the function to the variable
        paths = self.get_backpropagation_path(variable_id, function_id)
        for path in paths:
            for edge in path:
                edge.calculate_adjoint()
        
        return self.get_node(variable_id).get_adjoint_value()

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