from dolfinx import fem
import ufl
import scipy.sparse as sps

import dolfinx_adjoint.graph as graph

class NonlinearProblem(fem.petsc.NonlinearProblem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

class LinearProblem(fem.petsc.LinearProblem):
    # TODO under development
    def __init__(self, *args, **kwargs):
        if not "u" in kwargs:
            raise RuntimeError("LinearProblem requires a given solution in order to construct the graph correctly.")
        super().__init__(*args, **kwargs)
        graph.add_node(id(self), name="LinearProblem", tag="operation")
