import dolfinx_adjoint.graph as graph

from dolfinx import fem
import ufl

def form(*args, **kwargs):
    output = fem.form(*args, **kwargs)
    ufl_form = args[0]
    _node = graph.Adjoint(output)
    graph.add_node(id(output), name = ufl_form.__class__.__name__)

    graph.add_incoming_edges(id(output), ufl_form.coefficients())

    def adjoint(coefficient, argument = None):
        return fem.assemble_vector(fem.form(\
            ufl.derivative(ufl_form, coefficient, argument))).array[:]
    
    _node.set_adjoint_method(adjoint)
    _node.add_to_graph()

    return output