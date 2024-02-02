from dolfinx import fem

import scipy.sparse as sps

import dolfinx_adjoint.graph as graph

def dirichletbc(*args, map = None, **kwargs):
    if not "graph" in kwargs:
        output = fem.dirichletbc(*args, **kwargs)
        # output.dof_indices = args[1]
    else:
        _graph = kwargs["graph"]
        del kwargs["graph"]
        output = fem.dirichletbc(*args, **kwargs)
        # output.dof_indices = args[1]
        dofs = args[1]
        values = args[0]

        dirichletbc_node = graph.AbstractNode(output)
        _graph.add_node(dirichletbc_node)

        # Get node accociated with the value stored in args[0]
        value_node = _graph.get_node(id(args[0]))
        ctx = [dofs, values, map]
        dirichletbc_edge = DirichletBC_Edge(value_node, dirichletbc_node, ctx=ctx)
        dirichletbc_edge.set_next_functions(value_node.get_gradFuncs())
        dirichletbc_node.set_gradFuncs([dirichletbc_edge])
        _graph.add_edge(dirichletbc_edge) 
    return output

class DirichletBC_Edge(graph.Edge):
    def calculate_tlm(self):
        # Extract variables from contextvariable ctx
        dofs, _, map = self.ctx
        
        import numpy as np
        size = np.shape(self.input_value)[0]
        if np.shape(dofs)[0] == 2:
            dofs = dofs[0]
        matrix = sps.csr_matrix((np.ones(np.size(dofs)), (dofs, dofs)), (size, size))
        
        # If there exists a map from the function u to the function of the diricheltbc
        # we apply it. The map is stored as an array where the index is equivalent to the
        # index in the correct space and the value is the index in the wrong space
        if map == None:
            output = self.input_value @ matrix
        else:
            output = (self.input_value @ matrix)[map]
        
        # Convert to petsc vector
        from petsc4py import PETSc
        return PETSc.Vec().createWithArray(output)

