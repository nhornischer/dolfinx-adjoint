import dolfinx_adjoint.graph as graph

from dolfinx import fem
import ufl

def form(*args, **kwargs):
    output = fem.form(*args, **kwargs)
    ufl_form = args[0]
    graph.add_node(id(output), name = ufl_form.__class__.__name__)


    graph.add_incoming_edges(id(output), ufl_form.coefficients())
    graph.add_incoming_edges(id(output), ufl_form.constants())

    def adjoint_coefficient(coefficient, argument = None):
        ufl_dev = ufl.derivative(ufl_form, coefficient, argument)

        try:
            coefficient.dim == 0
            return fem.assemble_scalar(fem.form(ufl_dev))
        except:
            return fem.assemble_vector(fem.form(ufl_dev)).array[:]
    
    for coefficient in ufl_form.coefficients():
        _node = graph.Adjoint(output, coefficient)
        _node.set_adjoint_method(adjoint_coefficient)
        _node.add_to_graph()


    return output