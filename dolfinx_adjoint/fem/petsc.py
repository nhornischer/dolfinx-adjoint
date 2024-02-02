from dolfinx import fem, io
import ufl
import scipy.sparse as sps
import numpy as np
from mpi4py import MPI
import petsc4py as PETSc

import dolfinx_adjoint.graph as graph

from dolfinx.fem.petsc import NonlinearProblem as NonlinearProblemBase

class NonlinearProblem(NonlinearProblemBase):
    def __init__(self, *args, **kwargs):
        if not "graph" in kwargs:
            super().__init__(*args, **kwargs)
        else:
            _graph = kwargs["graph"]
            del kwargs["graph"]
            super().__init__(*args, **kwargs)

            F_form = args[0]
            u = args[1]

            problem_node = NonlinearProblemNode(self, F_form, u, **kwargs)
            _graph.add_node(problem_node)

            u_node = _graph.get_node(id(u))

            for coefficient in F_form.coefficients():
                if coefficient == u:
                    continue
                coefficient_node = _graph.get_node(id(coefficient))
                if not coefficient_node == None:
                    ctx = [F_form, u_node, coefficient, self.bcs, _graph]
                    coefficient_edge = NonlinearProblem_Coefficient_Edge(coefficient_node, problem_node, ctx=ctx)
                    _graph.add_edge(coefficient_edge)
                    problem_node.append_gradFuncs(coefficient_edge)
                    coefficient_edge.set_next_functions(coefficient_node.get_gradFuncs())
                    
            for constant in F_form.constants():
                constant_node = _graph.get_node(id(constant))
                if not constant_node == None:
                    ctx = [F_form, u_node, constant, self.bcs, _graph]
                    constant_edge = NonlinearProblem_Constant_Edge(constant_node, problem_node, ctx=ctx)
                    _graph.add_edge(constant_edge)
                    problem_node.append_gradFuncs(constant_edge)
                    constant_edge.set_next_functions(constant_node.get_gradFuncs())

            if hasattr(self, "bcs"):
                for bc in self.bcs:
                    bc_node = _graph.get_node(id(bc))
                    if not bc_node == None:
                        ctx = [F_form, u_node, self.bcs, self._a, _graph]
                        bc_edge = NonlinearProblem_Boundary_Edge(bc_node, problem_node, ctx=ctx)
                        _graph.add_edge(bc_edge)
                        problem_node.append_gradFuncs(bc_edge)
                        bc_edge.set_next_functions(bc_node.get_gradFuncs())

class NonlinearProblemNode(graph.AbstractNode):
    def __init__(self, object, F : ufl.form.Form, u : fem.Function, **kwargs):
        super().__init__(object)
        self.F = F
        self.u = u
        self.kwargs = kwargs

        self._name = "NonlinearProblem"

    def __call__(self):
        self.object = NonlinearProblem(self.F, self.u, self.kwargs)

class NonlinearProblem_Coefficient_Edge(graph.Edge):
    def calculate_adjoint(self):
        # Extract variables from contextvariable ctx
        F, u_node, m, bcs, _graph = self.ctx

        m_node = self.predecessor
        
        u = u_node.get_object()
        u_next = _graph.get_node(u_node.id, version = u_node.version + 1)
        
        # Construct the Jacobian J = ∂F/∂u
        V = u.function_space
        du = ufl.TrialFunction(V)
        F_manipulated = ufl.replace(F, {u: u_next.data, m: m_node.data})
        J = fem.petsc.assemble_matrix(fem.form(ufl.derivative(F_manipulated, u_node.data, du)), bcs = bcs)
        J.assemble()

        # Solve (J⁻¹)ᵀ λ = -x where x is the input with a sparse linear solver
        adjoint_solution = AdjointProblemSolver(J.transpose(), -self.input_value, fem.Function(V), bcs = bcs)

        # Calculate ∂F/∂m
        dFdm = fem.petsc.assemble_matrix(fem.form(ufl.derivative(F_manipulated, m_node.data)))
        dFdm.assemble()
        
        # Calculate λᵀ * ∂F/∂m
        return dFdm.transpose() * adjoint_solution.vector
    
class NonlinearProblem_Constant_Edge(graph.Edge):
    def calculate_adjoint(self):

        # Extract variables from contextvariable ctx
        F, u_node, m, bcs, _graph = self.ctx

        u = u_node.get_object()
        u_next = _graph.get_node(u_node.id, version = u_node.version + 1)
        
        # Construct the Jacobian J = ∂F/∂u
        V = u.function_space
        du = ufl.TrialFunction(V)
        J = fem.petsc.assemble_matrix(fem.form(ufl.derivative(F, u, du)), bcs = bcs)
        J.assemble()

        # Solve (J⁻¹)ᵀ λ = -x where x is the input with a sparse linear solver
        adjoint_solution = AdjointProblemSolver(J.transpose(), -self.input_value, fem.Function(V), bcs = bcs)

        # Create a function based on the constant in order to use ufl.derivative
        # to calculate ∂F/∂m
        domain = m.domain
        DG0 = fem.FunctionSpace(domain, ("DG", 0))
        function = fem.Function(DG0)
        function.vector.array[:] = m.c
        replaced_form = ufl.replace(F, {m: function})
        dFdm = fem.petsc.assemble_vector(fem.form(ufl.derivative(replaced_form, function)))
        dFdm.assemble()

        # Calculate λᵀ * ∂F/∂m
        # return x_np.T @ dFdm.array[:]
        return  adjoint_solution.vector.dot(dFdm)
    
class NonlinearProblem_Boundary_Edge(graph.Edge):
    def calculate_adjoint(self):

        # Extract variables from contextvariable ctx
        F, u_node, bcs, dFdbc_form, _graph = self.ctx

        u = u_node.get_object()
        u_next = _graph.get_node(u_node.id, version = u_node.version + 1)
        
        # Construct the Jacobian J = ∂F/∂u
        V = u.function_space
        du = ufl.TrialFunction(V)
        J = ufl.derivative(F, u, du)
        J = fem.petsc.assemble_matrix(fem.form(ufl.derivative(F, u, du)), bcs = bcs)
        J.assemble()

        # Solve (J⁻¹)ᵀ λ = -x where x is the input with a sparse linear solver
        adjoint_solution = AdjointProblemSolver(J.transpose(), -self.input_value, fem.Function(V), bcs = bcs)

        # ∂F/∂m = dFdbc defined in the nonlinear problem as a fem.Form
        dFdbc = fem.petsc.assemble_matrix(dFdbc_form)
        dFdbc.assemble()

        return dFdbc.transpose() * adjoint_solution.vector

def AdjointProblemSolver(A, b, x : fem.Function, bcs = None):
    from dolfinx import la
    from petsc4py import PETSc
    _x = la.create_petsc_vector_wrap(x.x)
    _solver = PETSc.KSP().create(x.function_space.mesh.comm)
    _solver.setOperators(A)

    _b = PETSc.Vec().createWithArray(b)
    from dolfinx.fem.petsc import set_bc
    set_bc(_b, bcs, scale=0.0)

    _solver.setType("preonly")
    _solver.getPC().setType("lu")
    _solver.getPC().setFactorSolverType("mumps")
    opts = PETSc.Options()
    opts["mat_mumps_icntl_24"] = 1  # Option to support solving a singular matrix (pressure nullspace)
    opts["mat_mumps_icntl_25"] = 0  # Option to support solving a singular matrix (pressure nullspace)
    opts["ksp_error_if_not_converged"] = 1
    _solver.setFromOptions()
    _solver.solve(_b, _x)

    return x
