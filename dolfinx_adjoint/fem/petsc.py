from dolfinx import fem
import ufl
import scipy.sparse as sps

import dolfinx_adjoint.graph as graph

class NonlinearProblem(fem.petsc.NonlinearProblem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        graph.add_node(id(self), name="NonlinearProblem")
        ufl_form = args[0]
        uh = args[1]


        dependencies = list(ufl_form.coefficients()) + list(ufl_form.constants())
        graph.add_incoming_edges(id(uh), dependencies)

        def adjoint(coefficient):
            dFdu = fem.assemble_matrix(fem.form(ufl.derivative(ufl_form, uh)), bcs=bcs)
            dFdu.finalize()
            V = uh.function_space
            shape = (V.dofmap.index_map.size_global, V.dofmap.index_map.size_global)
            dFduSparse = sps.csr_matrix((dFdu.data, dFdu.indices, dFdu.indptr), shape=shape)
            
            dFdf = fem.assemble_matrix(fem.form(ufl.derivative(ufl_form, coefficient))).to_dense()

            adjoint_solution = sps.linalg.spsolve(dFduSparse, -dFdf)

            return adjoint_solution.T
        
        for dependency in dependencies:
            _node = graph.Adjoint(uh, dependency)
            _node.set_adjoint_method(adjoint)
            _node.add_to_graph()

        try:
            bcs = kwargs["bcs"]
            for bc in bcs:
                graph.add_edge(id(bc), id(self))
        except: 
            pass