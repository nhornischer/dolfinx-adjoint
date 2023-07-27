from dolfinx import fem
from dolfinx.fem.function import Function
import dolfinx_adjoint.graph as graph


class Function(fem.Function):
    def __init__(self, *args, **kwargs):
        if not "name" in kwargs:
            kwargs["name"] = "Function"
        super().__init__(*args, **kwargs)
        graph.add_node(id(self), name=self.name, tag="data")

    def copy(self) -> Function:
        output = super().copy()
        graph.add_node(id(output), name=self.name + "copy", tag="operation")
        graph.add_edge(id(self), id(output))
        return output
    
    def interpolate(self, *args, **kwargs) -> Function:
        output = super().interpolate(*args, **kwargs)

        graph.add_node(id(output), name="interpolate", tag="operation")
        graph.add_edge(id(output), id(self))

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
        
