from dolfinx import fem

import scipy.sparse as sps

import dolfinx_adjoint.graph as graph

def dirichletbc(*args, **kwargs):
    output = fem.dirichletbc(*args, **kwargs)
    graph.add_node(id(output), name="DirichletBC")
    graph.add_edge(id(args[0]), id(output))
    dofs = args[1]
    values = args[0]
    def adjoint(*args):
        # Create diagonal matrix with ones where the boundary condition is applied
        # and zeros elsewhere
        import numpy as np
        indices_of_ones = dofs
        n = values.x.array.shape[0]
        matrix = sps.csr_matrix((np.ones(np.size(indices_of_ones)), (indices_of_ones, indices_of_ones)), (n, n))
        return matrix
    
    _node = graph.Adjoint(output, args[0])
    _node.set_adjoint_method(adjoint)
    _node.add_to_graph()
    
    
    return output