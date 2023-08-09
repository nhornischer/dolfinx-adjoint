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
        dofs = args[1]
        values = args[0]

        dirichletbc_node = graph.AbstractNode(output)
        _graph.add_node(dirichletbc_node)

        # Get node accociated with the value stored in args[0]
        value_node = _graph.get_node(id(args[0]))
        ctx = [dofs, values]
        dirichletbc_edge = DirichletBC_Edge(value_node, dirichletbc_node, ctx=ctx)
        dirichletbc_edge.set_next_functions(value_node.get_gradFuncs())
        dirichletbc_node.set_gradFuncs([dirichletbc_edge])
        _graph.add_edge(dirichletbc_edge) 
    return output

class DirichletBC_Edge(graph.Edge):
    def calculate_tlm(self):
        # Extract variables from contextvariable ctx
        dofs, values = self.ctx

        import numpy as np
        matrix = sps.csr_matrix((np.ones(np.size(dofs)), (dofs, dofs)), (values.x.array.shape[0], values.x.array.shape[0]))

        return self.input_value @ matrix
