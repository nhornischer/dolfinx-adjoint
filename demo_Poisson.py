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
# Define mesh and finite element space
domain = mesh.create_unit_square(MPI.COMM_WORLD, 64, 64, mesh.CellType.triangle)
V = fem.FunctionSpace(domain, ("CG", 1))                
W = fem.FunctionSpace(domain, ("DG", 0))

# Define the boundary and the boundary conditions
domain.topology.create_connectivity(domain.topology.dim -1, domain.topology.dim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
uD = fem.Function(V, name="u_D")                            
uD.interpolate(lambda x: 1.0 + 0.0 * x[0])                    
boundary_dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1, boundary_facets)
bc = fem.dirichletbc(uD, boundary_dofs)                     

# Define the basis functions and parameters
uh = fem.Function(V, name="uₕ")                             
v = ufl.TestFunction(V)
f = fem.Function(W, name="f")                               
nu = fem.Constant(domain, ScalarType(1.0), name = "ν")      
            
f.interpolate(lambda x: x[0] + x[1])                    

# Define the variational form and the residual equation
a = nu * ufl.inner(ufl.grad(uh), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx
F = a - L

# Define the problem solver and solve it
problem = fem.petsc.NonlinearProblem(F, uh, bcs=[bc])             
solver = nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)          
solver.solve(uh)                                                  
# Define profile g
g = fem.Function(W, name="g")                                   
g.interpolate(lambda x: 1 / (2 * np.pi**2) * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))   
# Define the objective function
alpha = fem.Constant(domain, ScalarType(1e-6), name = "α")      
J_form = 0.5 * ufl.inner(uh - g, uh - g) * ufl.dx + alpha * 0.5 *ufl.inner(f,f) * ufl.dx
J = fem.assemble_scalar(fem.form(J_form))                       
print("J(u) = ", J)

visualise()


# dJdf = compute_gradient(J, f)
# dJdnu = compute_gradient(J, nu)
# print("dJ/df =", dJdf)
# print("dJ/dnu =", dJdnu)

dJdbc = compute_gradient(J, bc)

dJdbc_function = dolfinx.fem.Function(V, name="dJdbc")
dJdbc_function.vector.setArray(dJdbc)


with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "demo_Poisson_dJdbc.xdmf", "w") as file:
    file.write_mesh(domain)
    file.write_function(dJdbc_function)

# dJdf_function = dolfinx.fem.Function(W, name="dJdf")
# dJdf_function.vector.setArray(dJdf)
# 
# 
# with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "demo_Poisson.xdmf", "w") as file:
#     file.write_mesh(domain)
#     file.write_function(uh)
# with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "demo_Poisson_dJdf.xdmf", "w") as file:
#     file.write_mesh(domain)
#     file.write_function(dJdf_function)



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