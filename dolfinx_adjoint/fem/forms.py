import dolfinx_adjoint.graph as graph

from dolfinx import fem
import ufl

def form(*args, **kwargs):
    output = fem.form(*args, **kwargs)
    ufl_form = args[0]
    _graph = graph.get_graph()
    _graph.add_node(id(output), name=ufl_form.__class__.__name__)
    for component in ufl_form.coefficients():
        graph.add_edge(id(component), id(output))

    def adjoint(coefficient, argument = None):
        return fem.assemble_vector(fem.form(ufl.derivative(ufl_form, coefficient, argument))).array[:]

    _graph.nodes[id(output)]["adjoint"] = adjoint

    
    return output