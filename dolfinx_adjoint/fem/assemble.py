import dolfinx_adjoint.graph as graph

from dolfinx import fem

def assemble_scalar(*args, **kwargs):
    output = fem.assemble_scalar(*args, **kwargs)
    graph.add_node(id(output), name="assemble_scalar")
    for i in range(len(args)):
        graph.add_edge(id(args[i]), id(output))
    _node = graph.Adjoint(output)
    _node.set_adjoint_method(lambda args: 1.0)
    _node.add_to_graph()
    return output

