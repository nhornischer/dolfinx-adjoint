from dolfinx import fem
import ufl
import scipy.sparse as sps
import numpy as np

import dolfinx_adjoint.graph as graph

class NonlinearProblem(fem.petsc.NonlinearProblem):
    def __init__(self, *args, **kwargs):
        if not "graph" in kwargs:
            super().__init__(*args, **kwargs)
        else:
            _graph = kwargs["graph"]
            del kwargs["graph"]
            super().__init__(*args, **kwargs)

            F_form = args[0]
            u = args[1]

            u_node = _graph.get_node(id(u))
            for coefficient in F_form.coefficients():
                if coefficient == u:
                    continue
                coefficient_node = _graph.get_node(id(coefficient))
                if not coefficient_node == None:
                    ctx = [F_form, u, coefficient, self.bcs]
                    coefficient_edge = NonlinearProblem_Coefficient_Edge(coefficient_node, u_node, ctx=ctx)
                    _graph.add_edge(coefficient_edge)
                    u_node.append_gradFuncs(coefficient_edge)
                    coefficient_edge.set_next_functions(coefficient_node.get_gradFuncs())
                    
            for constant in F_form.constants():
                constant_node = _graph.get_node(id(constant))
                if not constant_node == None:
                    ctx = [F_form, u, constant, self.bcs]
                    constant_edge = NonlinearProblem_Constant_Edge(constant_node, u_node, ctx=ctx)
                    _graph.add_edge(constant_edge)
                    u_node.append_gradFuncs(constant_edge)
                    constant_edge.set_next_functions(constant_node.get_gradFuncs())

            if hasattr(self, "bcs"):
                for bc in self.bcs:
                    bc_node = _graph.get_node(id(bc))
                    if not bc_node == None:
                        ctx = [F_form, u, self.bcs, self._a]
                        bc_edge = NonlinearProblem_Boundary_Edge(bc_node, u_node, ctx=ctx)
                        _graph.add_edge(bc_edge)
                        u_node.append_gradFuncs(bc_edge)
                        bc_edge.set_next_functions(bc_node.get_gradFuncs())


class NonlinearProblem_Coefficient_Edge(graph.Edge):
    def calculate_tlm(self):
        # Extract variables from contextvariable ctx
        F, u, m, bcs = self.ctx

        # Construct the Jacobian J = ∂F/∂u
        V = u.function_space
        du = ufl.TrialFunction(V)
        J = ufl.derivative(F, u, du)
        J = fem.assemble_matrix(fem.form(J), bcs = bcs)
        J.finalize()
        
        # Calculate ∂F/∂m
        dFdm = ufl.derivative(F, m)
        dFdm = fem.assemble_matrix(fem.form(dFdm))

        # Solve for ∂u/∂m = J⁻¹ ∂F/∂m with a sparse linear solver
        shape = (V.dofmap.index_map.size_global, V.dofmap.index_map.size_global)
        sparse_J = sps.csr_matrix((J.data, J.indices, J.indptr), shape=shape)
        dudm = sps.linalg.spsolve(sparse_J, -dFdm.to_dense())

        self.operator = dudm

        return self.input_value @ dudm
    
    def calculate_adjoint(self):
        # Extract variables from contextvariable ctx
        F, u, m, bcs = self.ctx

        # Construct the Jacobian J = ∂F/∂u
        V = u.function_space
        du = ufl.TrialFunction(V)
        J = ufl.derivative(F, u, du)
        J = fem.assemble_matrix(fem.form(J), bcs = bcs)
        J.finalize()

        # Solve (J⁻¹)ᵀ λ = -x where x is the input with a sparse linear solver
        shape = (V.dofmap.index_map.size_global, V.dofmap.index_map.size_global)
        sparse_J = sps.csr_matrix((J.data, J.indices, J.indptr), shape=shape)
        adjoint_solution = sps.linalg.spsolve(sparse_J.T, -self.input_value)

        # Calculate ∂F/∂m
        dFdm = ufl.derivative(F, m)
        dFdm = fem.assemble_matrix(fem.form(dFdm))

        # Calculate λᵀ * ∂F/∂m
        return adjoint_solution.T @ dFdm.to_dense()
    
class NonlinearProblem_Constant_Edge(graph.Edge):
    def calculate_tlm(self):
        
        # Extract variables from contextvariable ctx
        F, u, m, bcs = self.ctx

        # Construct the Jacobian J = ∂F/∂u
        V = u.function_space
        du = ufl.TrialFunction(V)
        J = ufl.derivative(F, u, du)
        J = fem.assemble_matrix(fem.form(J), bcs = bcs)
        J.finalize()

        # Create a function based on the constant in order to use ufl.derivative to
        # calculate ∂F/∂m
        domain = m.domain
        DG0 = fem.FunctionSpace(domain, "DG", 0)
        function = fem.Function(DG0)
        function.vector.array[:] = m.c
        replaced_form = ufl.replace(F, {m: function})
        dFdm = ufl.derivative(replaced_form, function)
        dFdm = fem.assemble_vector(fem.form(dFdm))

        # Solve for ∂u/∂m = J⁻¹ ∂F/∂m with a sparse linear solver
        shape = (V.dofmap.index_map.size_global, V.dofmap.index_map.size_global)
        sparse_J = sps.csr_matrix((J.data, J.indices, J.indptr), shape=shape)
        dudm = sps.linalg.spsolve(sparse_J, -dFdm.array[:])

        self.operator = dudm

        return self.input_value @ dudm
    
    def calculate_adjoint(self):

        # Extract variables from contextvariable ctx
        F, u, m, bcs = self.ctx

        # Construct the Jacobian J = ∂F/∂u
        V = u.function_space
        du = ufl.TrialFunction(V)
        J = ufl.derivative(F, u, du)
        J = fem.assemble_matrix(fem.form(J), bcs = bcs)
        J.finalize()

        # Solve (J⁻¹)ᵀ λ = -x where x is the input with a sparse linear solver
        shape = (V.dofmap.index_map.size_global, V.dofmap.index_map.size_global)
        sparse_J = sps.csr_matrix((J.data, J.indices, J.indptr), shape=shape)
        adjoint_solution = sps.linalg.spsolve(sparse_J.T, -self.input_value)

        # Create a function based on the constant in order to use ufl.derivative
        # to calculate ∂F/∂m
        domain = m.domain
        DG0 = fem.FunctionSpace(domain, ("DG", 0))
        function = fem.Function(DG0)
        function.vector.array[:] = m.c
        replaced_form = ufl.replace(F, {m: function})
        dFdm = ufl.derivative(replaced_form, function)
        dFdm = fem.assemble_vector(fem.form(dFdm))

        # Calculate λᵀ * ∂F/∂m
        return  adjoint_solution.T @ dFdm.array[:]
    
class NonlinearProblem_Boundary_Edge(graph.Edge):
    def calculate_tlm(self):

        # Extract variables from contextvariable ctx
        F, u, bcs, dFdbc_form, bc = self.ctx
            
        # Construct the Jacobian J = ∂F/∂u
        V = u.function_space
        du = ufl.TrialFunction(V)
        J = ufl.derivative(F, u, du)
        J = fem.assemble_matrix(fem.form(J), bcs = bcs)
        J.finalize()

        # ∂F/∂m = dFdbc defined in the nonlinear problem as a fem.Form
        dFdbc = fem.assemble_matrix(dFdbc_form)
        dFdbc.finalize()

        # Solve for ∂u/∂m = J⁻¹ ∂F/∂m with a sparse linear solver
        shape = (V.dofmap.index_map.size_global, V.dofmap.index_map.size_global)
        sparse_J = sps.csr_matrix((J.data, J.indices, J.indptr), shape=shape)
        dudm = sps.linalg.spsolve(sparse_J, -dFdbc.to_dense())

        self.operator = dudm

        return self.input_value @ dudm
    
    def calculate_adjoint(self):

        # Extract variables from contextvariable ctx
        F, u, bcs, dFdbc_form = self.ctx

        # Construct the Jacobian J = ∂F/∂u
        V = u.function_space
        du = ufl.TrialFunction(V)
        J = ufl.derivative(F, u, du)
        J = fem.assemble_matrix(fem.form(J), bcs = bcs)
        J.finalize()

        shape = (V.dofmap.index_map.size_global, V.dofmap.index_map.size_global)
        sparse_J = sps.csr_matrix((J.data, J.indices, J.indptr), shape=shape)
        adjoint_solution = sps.linalg.spsolve(sparse_J.T, -self.input_value)

        # ∂F/∂m = dFdbc defined in the nonlinear problem as a fem.Form
        dFdbc = fem.assemble_matrix(dFdbc_form)
        dFdbc.finalize()

        return adjoint_solution.T @ dFdbc.to_dense()
    