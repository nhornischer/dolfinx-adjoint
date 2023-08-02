from dolfinx import nls
import dolfinx_adjoint.graph as graph

class NewtonSolver(nls.petsc.NewtonSolver):
    def __init__(self, *args, **kwargs):
        if not "graph" in kwargs:
            super().__init__(*args, **kwargs)
        else:
            self._graph = kwargs["graph"]
            del kwargs["graph"]
            super().__init__(*args, **kwargs)

            # Add Node
            NewtonSolver_node = graph.Node(self)
            self._graph.add_node(NewtonSolver_node)

            # Add the edge from the problem to the NewtonSolver
            problem_node = self._graph.get_node(id(args[1]))
            NewtonSolver_edge = graph.Edge(problem_node, NewtonSolver_node)
            self._graph.add_edge(NewtonSolver_edge)
            
        graph.add_node(id(self), name="NewtonSolver")
        graph.add_edge(id(args[1]), id(self))

    def solve(self, *args, **kwargs):
        output = super().solve(*args, **kwargs)

        # Add the edge from the NewtonSolver to the Function
        if hasattr(self, "_graph"):
            function_node = self._graph.get_node(id(args[0]))
            NewtonSolver_node = self._graph.get_node(id(self))
            NewtonSolver_edge = graph.Edge(NewtonSolver_node, function_node)
            self._graph.add_edge(NewtonSolver_edge)

        # Add the edge from the Function to the NewtonSolver
        graph.add_edge(id(args[0]), id(self))
        return output
