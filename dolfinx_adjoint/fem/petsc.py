from dolfinx import fem
import ufl
import scipy.sparse as sps

import dolfinx_adjoint.graph as graph

class NonlinearProblem(fem.petsc.NonlinearProblem):
    def __init__(self, *args, **kwargs):
        if not "graph" in kwargs:
            super().__init__(*args, **kwargs)
        else:
            _graph = kwargs["graph"]
            del kwargs["graph"]
            super().__init__(*args, **kwargs)

            self.F_form = args[0]
            self.u = args[1]
            
            if "J" in kwargs:
                self.J = kwargs["J"]

            u_node = _graph.get_node(id(self.u))
            for coefficient in self.F_form.coefficients():
                if coefficient == self.u:
                    continue
                coefficient_node = _graph.get_node(id(coefficient))
                if not coefficient_node == None:
                    coefficient_edge = NonlinearProblem_Coefficient_Edge(coefficient_node, u_node, self)
                    _graph.add_edge(coefficient_edge)

            if hasattr(self, "bcs"):
                for bc in self.bcs:
                    bc_node = _graph.get_node(id(bc))
                    if not bc_node == None:
                        bc_edge = NonlinearProblem_Boundary_Edge(bc_node, u_node, self)
                        _graph.add_edge(bc_edge)


class NonlinearProblem_Coefficient_Edge(graph.Edge):
    def __init__(self, coefficient_node, u_node, nonlinear_problem):
        super().__init__(coefficient_node, u_node)
        self.F = nonlinear_problem.F_form
        self.u = nonlinear_problem.u
        self.bcs = nonlinear_problem.bcs
        if hasattr(nonlinear_problem, "J"):
            self.J = nonlinear_problem.J
        else:
            self.J = None

    def calculate_tlm(self):
        # Construct the Jacobian J = ∂F/∂u
        
        if not self.J == None:
            V = self.u.function_space
            du = ufl.TrialFunction(V)
            self.J = ufl.derivative(self.F, self.u, du)

        self.J = fem.assemble_matrix(fem.form(self.J), bcs = self.bcs)
        self.J.finalize()
        
        dFdcoefficient = ufl.derivative(self.F, self.predecessor.object)

        dFdcoefficient = fem.assemble_matrix(fem.form(dFdcoefficient))

        V = self.u.function_space
        shape = (V.dofmap.index_map.size_global, V.dofmap.index_map.size_global)

        
        sparse_J = sps.csr_matrix((self.J.data, self.J.indices, self.J.indptr), shape=shape)
        
        dudm = sps.linalg.spsolve(sparse_J, -dFdcoefficient.to_dense())

        self.tlm = dudm.T

        adjoint_successor = self.successor.get_adjoint_value()

        adjoint = fem.assemble_vector(fem.form(adjoint_successor))
        adjoint_predecessor = adjoint.array[:] @ dudm

        self.predecessor.set_adjoint_value(adjoint_predecessor)

        return dudm.T
    
    def calculate_adjoint(self):

        adjoint_successor = self.successor.get_adjoint_value()

        adjoint = fem.assemble_vector(fem.form(adjoint_successor))

        if not self.J == None:
            V = self.u.function_space
            du = ufl.TrialFunction(V)
            self.J = ufl.derivative(self.F, self.u, du)

        self.J = fem.assemble_matrix(fem.form(self.J), bcs = self.bcs)
        self.J.finalize()

        V = self.u.function_space
        shape = (V.dofmap.index_map.size_global, V.dofmap.index_map.size_global)

        sparse_J = sps.csr_matrix((self.J.data, self.J.indices, self.J.indptr), shape=shape)
    
        adjoint_value = sps.linalg.spsolve(sparse_J.T, -adjoint.array[:])
        
        dFdcoefficient = ufl.derivative(self.F, self.predecessor.object)
        dFdcoefficient = fem.assemble_matrix(fem.form(dFdcoefficient))

        predeccessor_adjoint_value = adjoint_value.T @ dFdcoefficient.to_dense()

        self.predecessor.set_adjoint_value(predeccessor_adjoint_value)
        return predeccessor_adjoint_value
    
class NonlinearProblem_Boundary_Edge(graph.Edge):
    def __init__(self, boundary_node, u_node, nonlinear_problem):
        super().__init__(boundary_node, u_node)
        self.nonlinear_problem = nonlinear_problem
        self.F = nonlinear_problem.F_form
        self.u = nonlinear_problem.u
        self.bcs = nonlinear_problem.bcs
        if hasattr(nonlinear_problem, "J"):
            self.J = nonlinear_problem.J
        else:
            self.J = None

    def calculate_tlm(self):
            
        # Construct the Jacobian J = ∂F/∂u
        if not self.J == None:
            V = self.u.function_space
            du = ufl.TrialFunction(V)
            self.J = ufl.derivative(self.F, self.u, du)

        self.J = fem.assemble_matrix(fem.form(self.J), bcs = self.bcs)
        self.J.finalize()

        print("dFdu", self.J.to_dense().shape, self.J.to_dense())
        
        dFdbc = fem.assemble_matrix(self.nonlinear_problem._a)
        dFdbc.finalize()

        print("dFdbc", dFdbc.to_dense().shape, dFdbc.to_dense())

        V = self.u.function_space
        shape = (V.dofmap.index_map.size_global, V.dofmap.index_map.size_global)

        sparse_J = sps.csr_matrix((self.J.data, self.J.indices, self.J.indptr), shape=shape)
        
        dudm = sps.linalg.spsolve(sparse_J, -dFdbc.to_dense())

        print("dudbc", dudm.shape, dudm)

        self.tlm = dudm.T

        adjoint_successor = self.successor.get_adjoint_value()

        adjoint = fem.assemble_vector(fem.form(adjoint_successor))
        adjoint_predecessor = adjoint.array[:] @ dudm

        self.predecessor.set_adjoint_value(adjoint_predecessor)

        return dudm.T
    
    def calculate_adjoint(self):

        adjoint_successor = self.successor.get_adjoint_value()

        adjoint = fem.assemble_vector(fem.form(adjoint_successor))

        if not self.J == None:
            V = self.u.function_space
            du = ufl.TrialFunction(V)
            self.J = ufl.derivative(self.F, self.u, du)

        self.J = fem.assemble_matrix(fem.form(self.J), bcs = self.bcs)
        self.J.finalize()

        V = self.u.function_space
        shape = (V.dofmap.index_map.size_global, V.dofmap.index_map.size_global)

        sparse_J = sps.csr_matrix((self.J.data, self.J.indices, self.J.indptr), shape=shape)
    
        adjoint_value = sps.linalg.spsolve(sparse_J.T, -adjoint.array[:])
        
        dFdbc = fem.assemble_matrix(self.nonlinear_problem._a)
        dFdbc.finalize()

        predeccessor_adjoint_value = adjoint_value.T @ dFdbc.to_dense()

        self.predecessor.set_adjoint_value(predeccessor_adjoint_value)
        return predeccessor_adjoint_value