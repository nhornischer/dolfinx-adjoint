import dolfinx_adjoint.graph as graph

from dolfinx import fem

def assemble_scalar(*args, **kwargs):
    output = fem.assemble_scalar(*args, **kwargs)
    _graph = graph.get_graph()
    _graph.add_node(id(output), name="assemble_scalar")
    for i in range(len(args)):
        graph.add_edge(id(args[i]), id(output))
    _graph.nodes[id(output)]["adjoint"] = lambda args: 1.0
    return output

def assemble_vector(*args, **kwargs):
    output = fem.assemble_vector(*args, **kwargs)
    return output

def assemble_matrix(*args, **kwargs):
    output = fem.assemble_matrix(*args, **kwargs)
    return output

