from dolfinx import fem
from dolfinx.fem.function import Function
import dolfinx_adjoint.graph as graph


class Function(fem.Function):
    def __init__(self, *args, **kwargs):
        if not "name" in kwargs:
            kwargs["name"] = "Function"
        if not "graph" in kwargs:
            super().__init__(*args, **kwargs)
        else:
            _graph = kwargs["graph"]
            del kwargs["graph"]
            super().__init__(*args, **kwargs)

            function_node = graph.Node(self, name=self.name)
            _graph.add_node(function_node)
        graph.add_node(id(self), name=self.name, tag="data")

    def copy(self) -> Function:
        output = super().copy()
        graph.add_node(id(output), name=self.name + "copy", tag="operation")
        graph.add_edge(id(self), id(output))
        return output
    
    
    
class Constant(fem.Function):
    def __init__(self, *args, **kwargs):
        domain, c = args
        if not "name" in kwargs:
            kwargs["name"] = "Constant"
        DG0 = fem.FunctionSpace(domain, ("DG", 0))
        super().__init__(DG0, **kwargs)

        self.vector.array[:] = c
        self.c = c
        self.dim = 0

        graph.add_node(id(self), name=self.name, tag = "data")
        
