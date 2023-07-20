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
        graph.add_node(id(ufl_form), name="residual")
        graph.add_attribute(id(ufl_form),'form', ufl_form)

        graph.add_edge(id(ufl_form), id(self))
        for component in ufl_form.coefficients():
            graph.add_edge(id(component), id(ufl_form))
        try:
            bcs = kwargs["bcs"]
            for bc in bcs:
                graph.add_edge(id(bc), id(self))
        except: 
            pass

        def adjoint(coefficient, argument = None):
            dFdu = fem.assemble_matrix(fem.form(ufl.derivative(ufl_form, uh)), bcs=bcs)
            dFdu.finalize()
            V = uh.function_space
            shape = (V.dofmap.index_map.size_global, V.dofmap.index_map.size_global)
            dFduSparse = sps.csr_matrix((dFdu.data, dFdu.indices, dFdu.indptr), shape=shape)
            
            dFdf = fem.assemble_matrix(fem.form(ufl.derivative(ufl_form, coefficient))).to_dense()

            adjoint_solution = sps.linalg.spsolve(dFduSparse, -dFdf)

            return adjoint_solution.T
        
        _node = graph.Adjoint(ufl_form)
        _node.set_adjoint_method(adjoint)
        _node.add_to_graph()