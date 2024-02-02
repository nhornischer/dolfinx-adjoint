from dolfinx.nls.petsc import NewtonSolver as NewtonSolverBase
import dolfinx_adjoint.graph as graph


class NewtonSolver(NewtonSolverBase):

    def __init__(self, *args, **kwargs):
        if "graph" not in kwargs:
            super().__init__(*args, **kwargs)
        else:
            _graph = kwargs["graph"]
            del kwargs["graph"]

            super().__init__(*args, **kwargs)

            solver_node = NewtonSolverNode(self, args[0], args[1])
            _graph.add_node(solver_node)

            problem_node = _graph.get_node(id(args[1]))

            if not problem_node == None:
                problem_edge = graph.Edge(problem_node, solver_node)
                solver_node.set_gradFuncs([problem_edge])
                _graph.add_edge(problem_edge)
                problem_edge.set_next_functions(problem_node.get_gradFuncs())

    def solve(self, *args, **kwargs):
        # Add the edge from the NewtonSolver to the Function
        if "graph" not in kwargs:
            output = super().solve(*args, **kwargs)
        else:
            if "version" in kwargs:
                version = kwargs["version"]
                del kwargs["version"]
            else:
                version = 1
            _graph = kwargs["graph"]
            del kwargs["graph"]

            solver_node = _graph.get_node(id(self))
            solve_node = SolveNode(args[0], solver_node, version = version, name = args[0].name)
            
            _graph.add_node(solve_node)
            if not solver_node == None:
                function_edge = graph.Edge(solver_node, solve_node)
                solve_node.set_gradFuncs([function_edge])
                _graph.add_edge(function_edge)
                function_edge.set_next_functions(solver_node.get_gradFuncs())

            output = super().solve(*args, **kwargs)

        return output
    

class NewtonSolverNode(graph.AbstractNode):
    def __init__(self, object,comm, problem, name = "NewtonSolver", **kwargs):
        super().__init__(object, name = name, **kwargs)
        self.comm = comm
        self.problem = problem    

    def __call__(self):
        self.object = NewtonSolverBase(self.comm, self.problem)

class SolveNode(graph.Node):
    def __init__(self, object, solverNode, name = "solve", **kwargs):
        super().__init__(object, name = name, **kwargs)
        self.solverNode = solverNode
        self.initial_values = object.vector.array.copy()
        

    def __call__(self):
        import numpy as np
        self.object.vector.array[:] = self.initial_values[:]
        self.solverNode.object.solve(self.object)