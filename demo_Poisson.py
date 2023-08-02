"""
In this demo we consider the parametric Poisson's equation with homogeneous Dirichlet boundary conditions:
    - ν ∆u = f              in Ω
         u = 0              on ∂Ω
where Ω = [0,1]² is the unit square, ν = 1 is the diffusion coefficient and f is a parameter.
We define the objective function (or quantity of interest)
    J(u) = ½∫_Ω ||u - g|| dx
where g is a known function.

The goal of this demo is to compute the gradient of J(u) with respect to f using the adjoint method.

We first start with the forward approach to solving the parametric Poisson's equation.
The weak formulation of the problem reads: find u ∈ V such that
    a(u, v) = L(v) ∀ v ∈ V
where
    a(u, v) = ∫_Ω ν ∇u · ∇v dx
    L(v)    = ∫_Ω f v dx.

We solve this problem with a residual equation
    F(u) = a(u, v) - L(v) != 0
"""

import numpy as np
import scipy.sparse as sps

from mpi4py import MPI
import dolfinx
from dolfinx import mesh, fem, io, nls
from dolfinx_adjoint import *
from petsc4py.PETSc import ScalarType
import ufl 

# We first need to create a graph object to store the computational graph. 
# This is done explicitly to maintain the guideline of FEniCSx.

graph_ = dolfinx_adjoint.graph.Graph()

# Define mesh and finite element space
domain = mesh.create_unit_square(MPI.COMM_WORLD, 64, 64, mesh.CellType.triangle)
V = fem.FunctionSpace(domain, ("CG", 1))                
W = fem.FunctionSpace(domain, ("DG", 0))

# Define the boundary and the boundary conditions
domain.topology.create_connectivity(domain.topology.dim -1, domain.topology.dim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
uD_L = fem.Function(V, name="u_D", graph=graph_)                          
uD_L.interpolate(lambda x: 1.0 + 0.0 * x[0])   
uD_R = fem.Function(V, name="u_D")
uD_R.interpolate(lambda x: 1.0 + 0.0 * x[0])         
uD_T = fem.Function(V, name="u_D")
uD_T.interpolate(lambda x: 1.0 + 0.0 * x[1])
uD_B = fem.Function(V, name="u_D")
uD_B.interpolate(lambda x: 1.0 + 0.0 * x[1])
        
boundary_dofs_L = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0))
boundary_dofs_R = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 1.0))
boundary_dofs_T = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[1], 1.0))
boundary_dofs_B = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[1], 0.0))

bc_L = fem.dirichletbc(uD_L, boundary_dofs_L, graph = graph_)     
bc_R = fem.dirichletbc(uD_R, boundary_dofs_R)   
bc_T = fem.dirichletbc(uD_T, boundary_dofs_T)
bc_B = fem.dirichletbc(uD_B, boundary_dofs_B)             

# Define the basis functions and parameters
uh = fem.Function(V, name="uₕ", graph = graph_)                             
v = ufl.TestFunction(V)
f = fem.Function(W, name="f", graph = graph_)                               
nu = fem.Constant(domain, ScalarType(1.0), name = "ν")      
            
f.interpolate(lambda x: x[0] + x[1])                    

# Define the variational form and the residual equation
a = nu * ufl.inner(ufl.grad(uh), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx
F = a - L

# Define the problem solver and solve it
problem = fem.petsc.NonlinearProblem(F, uh, bcs=[bc_L, bc_R, bc_T, bc_B], graph = graph_)             
solver = nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)          
solver.solve(uh)                                                  
# Define profile g
g = fem.Function(W, name="g")                                   
g.interpolate(lambda x: 1 / (2 * np.pi**2) * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))   
# Define the objective function
alpha = fem.Constant(domain, ScalarType(1e-6), name = "α")      
J_form = 0.5 * ufl.inner(uh - g, uh - g) * ufl.dx
J = fem.assemble_scalar(fem.form(J_form, graph = graph_), graph = graph_)                       
print("J(u) = ", J)
visualise()

# test = graph_.compute_adjoint(id(J), id(f))

# test_function= dolfinx.fem.Function(W, name="test")
# test_function.vector.setArray(test)

# with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "demo_Poisson_f.xdmf", "w") as file:
#     file.write_mesh(domain)
#     file.write_function(test_function)

# print(test.shape,test)

graph_.visualise("demo_Poisson_forward.pdf")
test = graph_.compute_adjoint(id(J), id(uD_L))

test_function= dolfinx.fem.Function(V, name="test")
test_function.vector.setArray(test)

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "demo_Poisson_bc.xdmf", "w") as file:
    file.write_mesh(domain)
    file.write_function(test_function)
graph_.visualise("demo_Poisson_forward.pdf")

# dJdf = compute_gradient(J, f)
# print("dJduh", dJdf.shape, dJdf)
# dJdnu = compute_gradient(J, nu)
dJdbc = compute_gradient(J, uD_L)

# print("dJ/dν", dJdnu)

# dJdf_function = dolfinx.fem.Function(W, name="dJdf")
# dJdf_function.vector.setArray(dJdf)
# dJdbc_function = dolfinx.fem.Function(V, name="dJdbc")
# dJdbc_function.vector.setArray(dJdbc)

# with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "demo_Poisson.xdmf", "w") as file:
#     file.write_mesh(domain)
#     file.write_function(uh)
# with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "demo_Poisson_dJdf.xdmf", "w") as file:
#     file.write_mesh(domain)
#     file.write_function(dJdf_function)
# with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "demo_Poisson_dJdbc.xdmf", "w") as file:
#     file.write_mesh(domain)
#     file.write_function(dJdbc_function)





import unittest

class TestPoisson(unittest.TestCase):
    def setUp(self) -> None:
        """
        We now turn to the explicit adjoint approach to calculate the gradient of J(u) with respect to f.
        Therefore, we first define a new reduced functional R(f) = J(u(f),f) and take the derivative with respect to f:
            dJ/df = dR/df = ∂J/∂u * du/df + ∂J/∂f                       (1)

        The first term ∂J/∂u and the last term ∂J/∂f are easy to compute.
        The second term du/df will be handled using the adjoint problem.

        By taking the derivative of F(u) = 0 with respect to f, we obtain a representation of du/df:
            dF/df = ∂F/∂u * du/df + ∂F/∂f = 0
            => du/df = -(∂F/∂u)^-1 * ∂F/∂f                              (2)

        Inserting (2) into (1) yields

            dJ/df = - ∂J/∂u * (∂F/∂u)^-1 * ∂F/∂f + ∂J/∂f                (3)

        Our adjoint approach thus is to solve the adjoint problem
            ∂Fᵀ/∂u * λ =  - ∂Jᵀ/∂u                                      (4)
        and then compute the gradient of J(u) with respect to f using (3).
            dJ/df = λᵀ * ∂F/∂f + ∂J/∂f                                  (5)   

        In our case ∂J/∂f = 0, so we only need to compute λᵀ * ∂F/∂f.
        """
        dFdu = fem.assemble_matrix(fem.form(ufl.derivative(F, uh)), bcs=[bc])
        dFdu.finalize()
        shape = (V.dofmap.index_map.size_global, V.dofmap.index_map.size_global)
        dFduSparse = sps.csr_matrix((dFdu.data, dFdu.indices, dFdu.indptr), shape=shape).transpose()

        dJdu = - fem.assemble_vector(fem.form(ufl.derivative(J_form, uh))).array.transpose()

        adjoint_solution = sps.linalg.spsolve(dFduSparse, dJdu)

        dFdf = fem.assemble_matrix(fem.form(ufl.derivative(F, f))).to_dense()

        dJdf = fem.assemble_vector(fem.form(ufl.derivative(J_form, f)))

        gradient = adjoint_solution.transpose() @ dFdf + dJdf.array

        self.dJ = fem.Function(W, name="dJdf")
        self.dJ.vector.setArray(gradient)
    

    def test_gradient(self):
        print("Testing gradient")
        self.assertTrue(np.allclose(compute_gradient(J, f), self.dJ.vector.array[:])) 