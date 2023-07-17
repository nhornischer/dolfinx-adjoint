from dolfinx import fem
import dolfinx_adjoint.graph as graph


class Function(fem.Function):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _graph = graph.get_graph()
        _graph.add_node(id(self), name=self.name)
        _graph.nodes[id(self)]["adjoint"] = lambda args: 1.0