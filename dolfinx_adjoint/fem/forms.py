import dolfinx_adjoint.graph as graph

from dolfinx import fem
import ufl

def form(*args, **kwargs):
    if not "graph" in kwargs:
        output = fem.form(*args, **kwargs)
    else:
        _graph = kwargs["graph"]
        del kwargs["graph"]
        output = fem.form(*args, **kwargs)
        form_node = graph.Node(output)
        _graph.add_node(form_node)

        ufl_form = args[0]
        ufl_node = graph.Node(ufl_form)
        _graph.add_node(ufl_node)

        form_edge = graph.Edge(ufl_node, form_node)
        _graph.add_edge(form_edge)

        for coefficient in ufl_form.coefficients():
            coefficient_node = _graph.get_node(id(coefficient))
            if not coefficient_node == None:
                coefficient_edge = FormEdge(coefficient_node, ufl_node)
                _graph.add_edge(coefficient_edge)
    return output


class FormEdge(graph.Edge):
    def calculate_tlm(self):
        derivative = ufl.derivative(self.successor.object, self.predecessor.object)

        self.tlm = derivative
        
        self.predecessor.set_adjoint_value(self.successor.get_adjoint_value() * derivative)

    def calculate_adjoint(self):
        return self.calculate_tlm()