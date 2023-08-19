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
        form_node = graph.AbstractNode(output)
        _graph.add_node(form_node)

        ufl_form = args[0]

        for coefficient in ufl_form.coefficients():
            coefficient_node = _graph.get_node(id(coefficient))
            if not coefficient_node == None:
                ctx = [ufl_form, coefficient]
                coefficient_edge = Form_Coefficient_Edge(coefficient_node, form_node, ctx=ctx)
                form_node.append_gradFuncs(coefficient_edge)
                coefficient_edge.set_next_functions(coefficient_node.get_gradFuncs())
                _graph.add_edge(coefficient_edge)

        for constant in ufl_form.constants():
            constant_node = _graph.get_node(id(constant))
            if not constant_node == None:
                ctx = [ufl_form, coefficient]
                coefficient_edge = Form_Constant_Edge(constant_node, form_node, ctx=ctx)
                form_node.append_gradFuncs(coefficient_edge)
                coefficient_edge.set_next_functions(constant_node.get_gradFuncs())
                _graph.add_edge(coefficient_edge)
    return output

class Form_Coefficient_Edge(graph.Edge):
    def calculate_tlm(self):
        ufl_form, coefficient = self.ctx
        derivative = ufl.derivative(ufl_form, coefficient)

        self.operator = derivative

        return fem.petsc.assemble_vector(fem.form(derivative))
    
class Form_Constant_Edge(graph.Edge):
    def calculate_tlm(self):
        ufl_form, constant = self.ctx

        # Create a function based on the constant in order to use ufl.derivative
        domain = constant.domain
        DG0 = fem.FunctionSpace(domain.mesh, ("DG", 0))
        function = fem.Function(DG0)
        function.vector.array[:] = constant.c

        replaced_form = ufl.replace(ufl_form, {constant: function})

        derivative = ufl.derivative(replaced_form, function)

        self.operator = derivative

        return self.input_value * fem.assemble_scalar(fem.form(derivative))