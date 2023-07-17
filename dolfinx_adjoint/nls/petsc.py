from dolfinx import nls
import dolfinx_adjoint.graph as graph

class NewtonSolver(nls.petsc.NewtonSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _graph = graph.get_graph()
        _graph.add_node(id(self), name="NewtonSolver")
        graph.add_edge(id(args[1]), id(self))

    def solve(self, *args, **kwargs):
        output = super().solve(*args, **kwargs)
        graph.add_edge(id(self), id(args[0]))

        return output