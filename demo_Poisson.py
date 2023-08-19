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
# Every function that is created with a graph object will be added to the graph
# and its gradient will be computed automatically. 

graph_ = graph.Graph()

# Define mesh and finite element space
domain = mesh.create_unit_square(MPI.COMM_WORLD, 64, 64, mesh.CellType.triangle)
V = fem.FunctionSpace(domain, ("CG", 1))                
W = fem.FunctionSpace(domain, ("DG", 0))

# Define the basis functions and parameters
uh = fem.Function(V, name="uₕ", graph=graph_)                             
v = ufl.TestFunction(V)
f = fem.Function(W, name="f", graph=graph_)                               
nu = fem.Constant(domain, ScalarType(1.0), name = "ν", graph = graph_)      
            
f.interpolate(lambda x: x[0] + x[1])

# Define the variational form and the residual equation
a = nu * ufl.inner(ufl.grad(uh), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx
F = a - L

# Define the boundary and the boundary conditions
domain.topology.create_connectivity(domain.topology.dim -1, domain.topology.dim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
uD_L = fem.Function(V, name="u_D", graph = graph_)                          
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

bcs = [fem.dirichletbc(uD_L, boundary_dofs_L, graph = graph_),
       fem.dirichletbc(uD_R, boundary_dofs_R),
       fem.dirichletbc(uD_T, boundary_dofs_T),
       fem.dirichletbc(uD_B, boundary_dofs_B)]
# Define the problem solver and solve it
problem = fem.petsc.NonlinearProblem(F, uh, bcs=bcs, graph = graph_)             
solver = nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)          
solver.solve(uh)                                                  
# Define profile g
g = fem.Function(W, name="g", graph = graph_)                                   
g.interpolate(lambda x: 1 / (2 * np.pi**2) * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))   
# Define the objective function
alpha = fem.Constant(domain, ScalarType(1e-6), name = "α")      
J_form = 0.5 * ufl.inner(uh - g, uh - g) * ufl.dx + alpha * ufl.inner(f, f) * ufl.dx
J = fem.assemble_scalar(fem.form(J_form, graph = graph_), graph = graph_)                       

graph_.visualise()
dJdf = graph_.backprop(id(J), id(f))
dJdnu = graph_.backprop(id(J), id(nu))
dJdbc = graph_.backprop(id(J), id(uD_L))

print("J(u) = ", J)
print("||dJ/df||_L2 = ", np.sqrt(np.dot(dJdf, dJdf)))
print("||dJ/dbc||_L2 = ", np.sqrt(np.dot(dJdbc, dJdbc)))
print("dJdnu = ", dJdnu)

# Visualise the results by saving them to a file as functions
dJdf_func = fem.Function(W, name="dJdf")
dJdf_func.vector.setArray(dJdf)
dJdbc_func = fem.Function(V, name="dJdbc")
dJdbc_func.vector.setArray(dJdbc)
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "poisson_results.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)
    xdmf.write_function(f)
    xdmf.write_function(g)
    xdmf.write_function(dJdf_func)
    xdmf.write_function(dJdbc_func)

import unittest

class TestPoisson(unittest.TestCase):
    def test_Poisson_dJdf(self):
        """
        We now turn to the explicit adjoint approach to calculate the gradient of J(u) with respect to f.
        Therefore, we first define a new reduced functional R(f) = J(u(f),f) and take the derivative with respect to f:
            dJ/df = dR/df = ∂J/∂u * du/df + ∂J/∂f                       (1.1)

        The first term ∂J/∂u and the last term ∂J/∂f are easy to compute.
        The second term du/df will be handled using the adjoint problem.

        By taking the derivative of F(u) = 0 with respect to f, we obtain a representation of du/df:
            dF/df = ∂F/∂u * du/df + ∂F/∂f = 0
            => du/df = -(∂F/∂u)^-1 * ∂F/∂f                              (1.2)

        Inserting (1.2) into (1.1) yields

            dJ/df = - ∂J/∂u * (∂F/∂u)^-1 * ∂F/∂f + ∂J/∂f                (1.3)

        Our adjoint approach thus is to solve the adjoint problem
            ∂Fᵀ/∂u * λ =  - ∂Jᵀ/∂u                                      (1.4)
        and then compute the gradient of J(u) with respect to f using (1.3).
            dJ/df = λᵀ * ∂F/∂f + ∂J/∂f                                  (1.5)   
        """
        dFdu = ufl.derivative(F, uh)

        dFdf = ufl.derivative(F, f)

        dJdu = ufl.derivative(J_form, uh)

        dJdf = ufl.derivative(J_form, f)

        dFdu = fem.assemble_matrix(fem.form(dFdu), bcs=bcs).to_dense()
        dJdu = fem.assemble_vector(fem.form(dJdu)).array
        dFdf = fem.assemble_matrix(fem.form(dFdf)).to_dense()
        dJdf = fem.assemble_vector(fem.form(dJdf)).array

        # We now need to apply the boundary conditions to the rhs of the adjoint problem
        for bc in bcs:
            for dofs in bc.dof_indices()[0]:
                dJdu[int(dofs)] = 0

        adjoint_solution = np.linalg.solve(dFdu.transpose(), - dJdu.transpose())
        
        gradient = adjoint_solution.transpose() @ dFdf + dJdf

        gradient_df= fem.Function(W, name="dJdf")
        gradient_df.vector.setArray(gradient)

        self.assertTrue(np.allclose(graph_.backprop(id(J), id(f)), gradient_df.vector.array[:]))

    def test_Poisson_dJdnu(self):
        """
        In this test we explicitly calculate the adjoint of J with respect to ν.
        Again we come up with
            dJ/dν = ∂J/∂u * du/dν + ∂J/∂ν                               (2.1)
        
        The first term ∂J/∂u is easy to compute, and to calculate the term ∂J/∂ν using a fem.function does the trick
        in order to use the automatic differentiation in ufl. Here we have to keep in mind that we still need 
        a scalar quantity as the value of ∂J/∂ν. 

        The second term du/dν will be handled using the adjoint problem similar to (1.2)
            dF/dν = ∂F/∂u * du/dν + ∂F/∂ν = 0
            => du/dν = -(∂F/∂u)^-1 * ∂F/∂ν                              (2.2)
        
        Inserting (2.2) into (2.1) yields
            dJ/dν = - ∂J/∂u * (∂F/∂u)^-1 * ∂F/∂ν + ∂J/∂ν                (2.3)
        
        Our adjoint approach thus is to solve the adjoint problem
            ∂Fᵀ/∂u * λ =  - ∂Jᵀ/∂u                                      (2.4)
        and then compute the gradient of J(u) with respect to ν using (2.3).
            dJ/dν = λᵀ * ∂F/∂ν + ∂J/∂ν                                  (2.5)
        """
        DG0 = fem.FunctionSpace(domain, ("DG", 0))
        nu_function = fem.Function(DG0, name="nu")
        nu_function.vector.array[:] = ScalarType(1.0)

        J_form_replaced = ufl.replace(J_form, {nu: nu_function})
        F_replaced = ufl.replace(F, {nu: nu_function})


        dJdu = ufl.derivative(J_form, uh)
        dJdnu = ufl.derivative(J_form_replaced, nu_function)
        dFdnu = ufl.derivative(F_replaced, nu_function)
        dFdu = ufl.derivative(F, uh)

        dJdu = fem.assemble_vector(fem.form(dJdu)).array
        dJdnu = fem.assemble_scalar(fem.form(dJdnu))
        dFdnu = fem.assemble_vector(fem.form(dFdnu)).array
        dFdu = fem.assemble_matrix(fem.form(dFdu), bcs=bcs).to_dense()

        # We now need to apply the boundary conditions to the rhs of the adjoint problem
        for bc in bcs:
            for dofs in bc.dof_indices()[0]:
                dJdu[int(dofs)] = 0
        
        adjoint_solution = np.linalg.solve(dFdu.transpose(), - dJdu.transpose())

        gradient = adjoint_solution.transpose() @ dFdnu + dJdnu

        self.assertTrue(np.allclose(graph_.backprop(id(J), id(nu)), gradient))
 
    def test_Poisson_dJdbc(self):
        """
        We now turn to the explicit adjoint approach to calculate the gradient of J(u) with respect to u_D on the left boudnary.
        Therefore, we first define a new reduced functional R(u_D) = J(u(u_D),u_D) and take the derivative with respect to u_D:
            dJ/du_D = ∂J/∂u * du/du_D + ∂J/∂u_D                         (3.1)

        The first term ∂J/∂u is easy to compute and for the second term we use the adjoint problem similar to (1.2)
            dF/du_D = ∂F/∂u * du/du_D + ∂F/∂u_D = 0
            => du/du_D = -(∂F/∂u)^-1 * ∂F/∂u_D                          (3.2)
        
        Inserting (3.2) into (3.1) yields
            dJ/du_D = - ∂J/∂u * (∂F/∂u)^-1 * ∂F/∂u_D + ∂J/∂u_D          (3.3)

        We first need to look closer to the term ∂F/∂u_D. 

        Since the boundary condition is not directly in the ufl representation of the form F,
        we need to look at out the boundary conditions are applied when constructing a nonlinear problem.
        In the case of the Poisson problem, the boundary conditions are applied to the linearized problem,
        by lifting the linear term. With the form F = a - L = 0, we get the assembled form F = A*u - b = 0.
        The boundary conditions are now applied by lifting the b vector:
            b = b - α A (u_D - x₀)
        where α is a scaling factor (in this case α=-1) and x_0 is some weird thing that is not important here.
        The term ∂F/∂u_D is now given by
            ∂F/∂u_D = - α A

        Our adjoint approach thus is to solve the adjoint problem
            ∂Fᵀ/∂u * λ =  - ∂Jᵀ/∂u                                      (3.4)
        and then compute the gradient of J(u) with respect to u_D using (3.3).
            dJ/du_D = λᵀ * ∂F/∂u_D + ∂J/∂u_D                            (3.5)

        Here ∂J/∂u_D is not defined. Since the function on the boundary u_D is defined for Ω and not for ∂Ω,
        especially not the part of the boundary where the boundary condition is applied. We need to extract these values
        and set everything else to zero.
        """

        dFdu = ufl.derivative(F, uh)
        dJdu = ufl.derivative(J_form, uh)

        dFdu = fem.assemble_matrix(fem.form(dFdu), bcs=bcs).to_dense()
        dJdu = fem.assemble_vector(fem.form(dJdu)).array
        dFdbc = fem.assemble_matrix(problem._a).to_dense()

        # We now need to apply the boundary conditions to the rhs of the adjoint problem
        for bc in bcs:
            for dofs in bc.dof_indices()[0]:
                dJdu[int(dofs)] = 0

        adjoint_solution = np.linalg.solve(dFdu.transpose(), - dJdu.transpose())
        
        gradient = adjoint_solution.transpose() @ dFdbc

        matrix = np.zeros((len(gradient), len(gradient)))
        for index in boundary_dofs_L:
            matrix[index, index] = 1.0

        gradient = matrix @ gradient

        gradient_bc = fem.Function(V, name="dJdbc")
        gradient_bc.vector.setArray(gradient)

        self.assertTrue(np.allclose(graph_.backprop(id(J), id(uD_L)), gradient_bc.vector.array[:]))

