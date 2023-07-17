from dolfinx import fem
import ufl
import scipy.sparse as sps

import dolfinx_adjoint.graph as graph

class NonlinearProblem(fem.petsc.NonlinearProblem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _graph = graph.get_graph()
        _graph.add_node(id(self), name="NonlinearProblem")
        ufl_form = args[0]
        uh = args[1]
        bcs = kwargs["bcs"]
        _graph.add_node(id(ufl_form), name="residual")
        _graph.nodes[id(ufl_form)]['form'] = ufl_form

        def adjoint(coefficient, argument = None):
            dFdu = fem.assemble_matrix(fem.form(ufl.derivative(ufl_form, uh)), bcs=bcs)
            dFdu.finalize()
            V = uh.function_space
            shape = (V.dofmap.index_map.size_global, V.dofmap.index_map.size_global)
            dFduSparse = sps.csr_matrix((dFdu.data, dFdu.indices, dFdu.indptr), shape=shape).transpose()
            
            dFdf = fem.assemble_matrix(fem.form(ufl.derivative(ufl_form, coefficient))).to_dense()

            adjoint_solution = sps.linalg.spsolve(dFduSparse, -dFdf)

            return adjoint_solution.T
        _graph.nodes[id(ufl_form)]["adjoint"] = adjoint

        graph.add_edge(id(ufl_form), id(self))
        for component in ufl_form.coefficients():
            graph.add_edge(id(component), id(ufl_form))
        graph.add_edge(id(kwargs["bcs"][0]), id(self))