from dolfinx.fem.petsc import NewtonSolverNonlinearProblem
from dolfinx.nls.petsc import NewtonSolver as NewtonSolverBase
from mpi4py import MPI

import dolfinx_adjoint.graph as graph


class NewtonSolver(NewtonSolverBase):
    """OVERLOADS: :py:class:`dolfinx.nls.petsc.NewtonSolver`
    A Newton solver for non-linear problems.

    The overloaded class modifies the initialization of the NewtonSolver to keep track of the dependencies
    in the computational graph and the adjoint equations. The original functionality is kept.

    """

    def __init__(self, *args, **kwargs):
        """OVERLOADS: :py:func:`dolfinx.nls.petsc.NewtonSolver.__init__`
        Initialize the Newton solver.

        Args:
            args: Arguments to :py:func:`dolfinx.nls.petsc.NewtonSolver.__init__`
            kwargs: Keyword arguments to :py:func:`dolfinx.nls.petsc.NewtonSolver.__init__`
            graph (graph, optional): An additional keyword argument to specifier whether the assemble
                operation should be added to the graph. If not present, the original functionality
                of dolfinx is used without any additional functionalities.

        """
        if "graph" not in kwargs:
            super().__init__(*args, **kwargs)
        else:
            _graph = kwargs["graph"]
            del kwargs["graph"]
            super().__init__(*args, **kwargs)

            solver_node = NewtonSolverNode(self, args[0], args[1])
            _graph.add_node(solver_node)

            # Creating and adding the edge to the graph
            problem_node = _graph.get_node(id(args[1]))
            if not problem_node == None:
                problem_edge = graph.Edge(problem_node, solver_node)
                solver_node.set_gradFuncs([problem_edge])
                _graph.add_edge(problem_edge)
                problem_edge.set_next_functions(problem_node.get_gradFuncs())

    def solve(self, *args, **kwargs):
        """OVERLOADS: :py:func:`dolfinx.nls.petsc.NewtonSolver.solve`
        Solve non-linear problem into function u. Returns the number of iterations and if the solver converged.

        Args:
            args: Arguments to :py:func:`dolfinx.nls.petsc.NewtonSolver.solve`
            kwargs: Keyword arguments to :py:func:`dolfinx.nls.petsc.NewtonSolver.solve`
            graph (graph, optional): An additional keyword argument to specifier whether the assemble
                operation should be added to the graph. If not present, the original functionality
                of dolfinx is used without any additional functionalities.
            version (int, optional): An additional keyword argument to specify the version of the solution node. If not present, the version is set to 1.

        Returns:
            int: Number of iterations
            bool: Convergence of the solver

        """

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
            solve_node = SolveNode(
                args[0], solver_node, version=version, name=args[0].name
            )
            _graph.add_node(solve_node)

            # Creating and adding the edge to the graph
            if not solver_node == None:
                function_edge = graph.Edge(solver_node, solve_node)
                solve_node.set_gradFuncs([function_edge])
                _graph.add_edge(function_edge)
                function_edge.set_next_functions(solver_node.get_gradFuncs())

            output = super().solve(*args, **kwargs)

        return output


class NewtonSolverNode(graph.AbstractNode):
    """
    Node for the initialization of :py:class:`dolfinx.nls.petsc.NewtonSolver`

    """

    def __init__(
        self,
        object: object,
        comm: MPI.Intracomm,
        problem: NewtonSolverNonlinearProblem,
        name="NewtonSolver",
        **kwargs,
    ):
        """
        Constructor for the NewtonSolverNode

        In order to create the NewtonSolver in the forward pass,
        the problem and the MPI communicator are needed.

        Args:
            object (object): The object to be wrapped in the node
            comm (MPI.Intracomm): The MPI communicator
            problem (dolfinx.fem.petsc.NewtonSolverNonlinearProblem): The non-linear problem to be solved
            name (str, optional): The name of the node
            kwargs (optional): Additional keyword arguments to be passed to the :py:class:`dolfinx_adjoint.graph.AbstractNode` constructor

        """
        super().__init__(object, name=name, **kwargs)
        self.comm = comm
        self.problem = problem

    def __call__(self):
        """
        The call method to initialize the NewtonSolver

        """
        self.object = NewtonSolverBase(self.comm, self.problem)


class SolveNode(graph.Node):
    """
    Node for the operation :py:func:`dolfinx.nls.petsc.NewtonSolver.solve`

    """

    def __init__(self, object: object, solverNode: graph.Node, name="solve", **kwargs):
        """
        Constructor for the SolveNode

        In order to solve the non-linear problem associated with the NewtonSolver,
        the solverNode storing the NewtonSolver and the non-linear problem are needed.

        Args:
            object (object): The object to be wrapped in the node
            solverNode (graph.Node): The node storing the NewtonSolver
            name (str, optional): The name of the node
            kwargs (optional): Additional keyword arguments to be passed to the :py:class:`dolfinx_adjoint.graph.AbstractNode` constructor

        """
        super().__init__(object, name=name, **kwargs)
        self.solverNode = solverNode
        self.initial_values = object.x.array.copy()

    def __call__(self):
        """
        The call method to solve the non-linear problem associated with the NewtonSolver

        """

        self.object.x.array[:] = self.initial_values[:]
        self.solverNode.object.solve(self.object)
