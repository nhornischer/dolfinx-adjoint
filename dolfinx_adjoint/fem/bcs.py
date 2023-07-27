from dolfinx import fem

import dolfinx_adjoint.graph as graph

def dirichletbc(*args, **kwargs):
    output = fem.dirichletbc(*args, **kwargs)
    graph.add_node(id(output), name="DirichletBC")
    graph.add_edge(id(args[0]), id(output))

    def adjoint(*args):
        return 1.0
    
    _node = graph.Adjoint(output, args[0])
    _node.set_adjoint_method(adjoint)
    _node.add_to_graph()
    
    
    return output