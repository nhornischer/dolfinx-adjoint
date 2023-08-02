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

        graph.add_node(id(self), name="NonlinearProblem", tag="operation")
        ufl_form = args[0]
        uh = args[1]

        dependencies_coefficients = list(ufl_form.coefficients())
        dependencies_constants = list(ufl_form.constants())
        graph.add_incoming_edges(id(self), dependencies_coefficients)
        graph.add_incoming_edges(id(self), dependencies_constants)

        def adjoint_coefficient(coefficient):
            dFdu = fem.assemble_matrix(fem.form(ufl.derivative(ufl_form, uh)), bcs=bcs)
    
            dFdu.finalize()
            print("dFdu", dFdu.to_dense().shape, dFdu.to_dense())
            V = uh.function_space
            shape = (V.dofmap.index_map.size_global, V.dofmap.index_map.size_global)
            dFduSparse = sps.csr_matrix((dFdu.data, dFdu.indices, dFdu.indptr), shape=shape)

            dFdm_form = fem.form(ufl.derivative(ufl_form, coefficient))
            try:
                coefficient.dim == 0
                dFdm = fem.assemble_vector(dFdm_form).array[:]
            except:
                dFdm = fem.assemble_matrix(dFdm_form).to_dense()
            print("dFdm", dFdm.shape, dFdm)
            
            dudm = sps.linalg.spsolve(dFduSparse, -dFdm)

            print("dudm",dudm.shape, dudm)

            return dudm.T
        
        try:
            dependencies_coefficients.pop(dependencies_coefficients.index(uh))
        except:
            pass
        for dependency in dependencies_coefficients:
            _node = graph.Adjoint(uh, dependency, "implicit")
            _node.set_adjoint_method(adjoint_coefficient)
            _node.add_to_graph()

        def adjoint_boundary_values(bc):
            # TODO: under development
            dFdu = fem.assemble_matrix(fem.form(ufl.derivative(ufl_form, uh)), bcs=bcs)
            dFdu.finalize()
            V = uh.function_space
            shape = (V.dofmap.index_map.size_global, V.dofmap.index_map.size_global)
            dFduSparse = sps.csr_matrix((dFdu.data, dFdu.indices, dFdu.indptr), shape=shape)
            print("dFdu", dFdu.to_dense().shape, dFdu.to_dense())

            A = fem.assemble_matrix(self._a)
            A.finalize()
            A = A.to_dense()
            dFdbc = A
            print("dFdbc",dFdbc.shape, dFdbc)

            dudbc = sps.linalg.spsolve(dFduSparse, -dFdbc)

            print("dudbc",dudbc.shape, dudbc)

            return dudbc.T

        try:
            bcs = kwargs["bcs"]
            # TODO: add adjoint for boundary values
            for bc in bcs:
                graph.add_edge(id(bc), id(self))
                _node = graph.Adjoint(uh, bc, "implicit")
                _node.set_adjoint_method(adjoint_boundary_values)
                _node.add_to_graph()

        except: 
            pass


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