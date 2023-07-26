from dolfinx import fem
from dolfinx.fem.function import Function
import dolfinx_adjoint.graph as graph


class Function(fem.Function):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        graph.add_node(id(self), name=self.name, tag="data")

    def copy(self) -> Function:
        output = super().copy()
        graph.add_node(id(output), name=self.name + "copy", tag="operation")
        graph.add_edge(id(self), id(output))

        return output
    
class Constant(fem.Constant):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        graph.add_node(id(self), name="Constant", tag= "data")
