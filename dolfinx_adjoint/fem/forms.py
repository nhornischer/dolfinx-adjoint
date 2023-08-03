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
                coefficient_edge = Form_Coefficient_Edge(coefficient_node, ufl_node)
                _graph.add_edge(coefficient_edge)

        for constant in ufl_form.constants():
            constant_node = _graph.get_node(id(constant))
            if not constant_node == None:
                constant_edge = Form_Constant_Edge(constant_node, ufl_node)
                _graph.add_edge(constant_edge)
    return output

class Form_Coefficient_Edge(graph.Edge):
    def calculate_tlm(self):
        derivative = ufl.derivative(self.successor.object, self.predecessor.object)

        self.tlm = derivative
        
        self.predecessor.set_adjoint_value(self.successor.get_adjoint_value() * derivative)

    def calculate_adjoint(self):
        return self.calculate_tlm()
    
class Form_Constant_Edge(graph.Edge):
    def calculate_tlm(self):
        form = self.successor.object
        constant = self.predecessor.object

        # Create a function based on the constant in order to use ufl.derivative
        domain = constant.domain
        DG0 = fem.FunctionSpace(domain.mesh, ("DG", 0))
        function = fem.Function(DG0)
        function.vector.array[:] = constant.c

        replaced_form = ufl.replace(form, {constant: function})

        derivative = ufl.derivative(replaced_form, function)

        self.tlm = derivative
        
        self.predecessor.set_adjoint_value(self.successor.get_adjoint_value() * derivative)

    def calculate_adjoint(self):
        return self.calculate_tlm()