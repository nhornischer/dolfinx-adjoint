import dolfinx_adjoint.graph as graph

from dolfinx import fem
import ufl

def form(*args, **kwargs):
    output = fem.form(*args, **kwargs)
    ufl_form = args[0]
    graph.add_node(id(output), name = ufl_form.__class__.__name__)

    dependencies = list(ufl_form.coefficients()) + list(ufl_form.constants())

    graph.add_incoming_edges(id(output), dependencies)

    def adjoint(coefficient, argument = None):
        return fem.assemble_vector(fem.form(ufl.derivative(ufl_form, coefficient, argument))).array[:]
    
    for dependency in dependencies:
        _node = graph.Adjoint(output, dependency)
        _node.set_adjoint_method(adjoint)
        _node.add_to_graph()

    return output