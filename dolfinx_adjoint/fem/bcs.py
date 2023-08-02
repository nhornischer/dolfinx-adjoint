from dolfinx import fem

import scipy.sparse as sps

import dolfinx_adjoint.graph as graph

def dirichletbc(*args, **kwargs):
    if not "graph" in kwargs:
        output = fem.dirichletbc(*args, **kwargs)
    else:
        _graph = kwargs["graph"]
        del kwargs["graph"]
        output = fem.dirichletbc(*args, **kwargs)
        output.dofs = args[1]
        output.values = args[0]

        dirichletbc_node = graph.Node(output)
        _graph.add_node(dirichletbc_node)

        # Get node accociated with the value stored in args[0]
        value_node = _graph.get_node(id(args[0]))
        dirichletbc_edge = DirichletBC_Edge(value_node, dirichletbc_node)
        _graph.add_edge(dirichletbc_edge) 
    return output

class DirichletBC_Edge(graph.Edge):
    def calculate_tlm(self):
        adjoint_value = self.successor.get_adjoint_value()

        import numpy as np
        matrix = sps.csr_matrix((np.ones(np.size(self.successor.object.dofs)), (self.successor.object.dofs, self.successor.object.dofs)), (self.successor.object.values.x.array.shape[0], self.successor.object.values.x.array.shape[0]))

        adjoint_value = adjoint_value @ matrix

        self.predecessor.set_adjoint_value(adjoint_value)

    def calculate_adjoint(self):
        return self.calculate_tlm()
