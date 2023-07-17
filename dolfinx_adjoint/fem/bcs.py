from dolfinx import fem

import dolfinx_adjoint.graph as graph

def dirichletbc(*args, **kwargs):
    output = fem.dirichletbc(*args, **kwargs)
    _graph = graph.get_graph()
    _graph.add_node(id(output), name="DirichletBC")
    graph.add_edge(id(args[0]), id(output))
    return output