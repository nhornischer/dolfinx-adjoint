import dolfinx_adjoint.graph as graph
from dolfinx import fem

def assemble_scalar(*args, **kwargs):
    if not "graph" in kwargs:
        output = fem.assemble_scalar(*args, **kwargs)
    else:
        _graph = kwargs["graph"]
        del kwargs["graph"]
        output = fem.assemble_scalar(*args, **kwargs)

        # Creating and adding node to graph
        assemble_node = graph.Node(output, name="assemble_scalar")
        _graph.add_node(assemble_node)

        # Create edge between form and assemble
        form_node = _graph.get_node(id(args[0]))
        assemble_edge = graph.Edge(form_node, assemble_node)
        assemble_node.set_gradFuncs([assemble_edge])

        # Create connectivity to previous edges
        assemble_edge.set_next_functions(form_node.get_gradFuncs())
        _graph.add_edge(assemble_edge)

    return output