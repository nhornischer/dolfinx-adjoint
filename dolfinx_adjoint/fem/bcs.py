from dolfinx import fem

import dolfinx_adjoint.graph as graph

def dirichletbc(*args, **kwargs):
    output = fem.dirichletbc(*args, **kwargs)
    graph.add_node(id(output), name="DirichletBC")
    graph.add_edge(id(args[0]), id(output))

    def adjoint():
        return None
    
    
    return output