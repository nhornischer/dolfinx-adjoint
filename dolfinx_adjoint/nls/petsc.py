from dolfinx import nls
import dolfinx_adjoint.graph as graph

class NewtonSolver(nls.petsc.NewtonSolver):

    def __init__(self, *args, **kwargs):
        if "graph" not in kwargs:
            super().__init__(*args, **kwargs)
        else:
            _graph = kwargs["graph"]
            del kwargs["graph"]

            super().__init__(*args, **kwargs)

            solver_node = graph.AbstractNode(self)
            _graph.add_node(solver_node)

            problem_node = _graph.get_node(id(args[1]))

            problem_edge = graph.Edge(problem_node, solver_node)
            solver_node.set_gradFuncs([problem_edge])
            _graph.add_edge(problem_edge)
            problem_edge.set_next_functions(problem_node.get_gradFuncs())

    def solve(self, *args, **kwargs):
        # Add the edge from the NewtonSolver to the Function
        if "graph" not in kwargs:
            output = super().solve(*args, **kwargs)
        else:
            function_node = graph.Node(args[0], version = 1, name = args[0].name)
            _graph = kwargs["graph"]
            del kwargs["graph"]
            _graph.add_node(function_node)

            solver_node = _graph.get_node(id(self))
            function_edge = graph.Edge(solver_node, function_node)
            function_node.set_gradFuncs([function_edge])
            _graph.add_edge(function_edge)
            function_edge.set_next_functions(solver_node.get_gradFuncs())

            output = super().solve(*args, **kwargs)

        return output
