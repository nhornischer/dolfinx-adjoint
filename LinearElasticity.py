"""
This example demonstrates the capabilities of the framework in modern engineering simulations by investigating the deformation of elastic bodies.
The governing equations for these problems are given by the equations of linear elasticity. 

We define the problem on a 3D domain Ω representing a beam with length L and witdh W.
The beam is clamped on the left side of the domain and the body force is acting on the beam.

Small elastic deformations of the body Ω under isotropic elastic conditions can be described by the displacement vector field u(x) : Ω → R^3 with the equations

     -∇⋅σ(u) = f in Ω
        σ(u) = λ tr(ε(u))I + 2με(u)
        ε(u) = 1/2(∇u + ∇uᵀ)

where σ(u) is the stress tensor, ε(u) is the strain tensor, λ and μ are the Lamé parameters, f is the body force per unit volume and I is the identity tensor.

We define the weak form of the linear elasticity problem on a vector-valued ansatz space V:
Find u ∈ V such that
    a(u, v) = L(v) for all v ∈ V
where
    a(u, v) = ∫_Ω σ(u) : ε(v) dx
    L(v) = ∫_Ω f ⋅ v dx + ∫_Γ T ⋅ v ds

In this form ε(v) is the symmetric part of ∇v and T is the traction vector at the boundary.

We solve this problem using the residual equation
    F(u) = a(u, v) - L(v) != 0

"""
import numpy as np
import ufl

from mpi4py import MPI
from petsc4py.PETSc import ScalarType

from dolfinx import mesh, fem, plot, io, nls
from dolfinx_adjoint import *
from basix.ufl import element
import os

dir = os.path.dirname(__file__)
graph_ = graph.Graph()

# Scaled variable
L = 1
W = 0.1
rho = 1
delta = W/L
gamma = 0.4*delta**2
g = gamma

domain = mesh.create_box(MPI.COMM_WORLD, [np.array([0,0,0]), np.array([L, W, W])],
                  [30,10,10], cell_type=mesh.CellType.hexahedron)

vector_element = element("Lagrange", domain.basix_cell(), 1, shape= (3,))
V = fem.FunctionSpace(domain, vector_element)
ds = ufl.Measure("ds", domain=domain)

u = fem.Function(V, name = "Deformation", graph = graph_)
v = ufl.TestFunction(V)

f = fem.Constant(domain, ScalarType((0, 0, -rho*g)))
T = fem.Constant(domain, ScalarType((0, 0, 0)))
lambda_ = fem.Constant(domain, ScalarType(1.0), graph = graph_)
mu = fem.Constant(domain, ScalarType(1.25), graph = graph_)

a = ufl.inner(lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2*mu*ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.dx
L = ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * ds
F = a - L

# Boundary condition describing the clamped left side of the beam
u_D = np.array([0,0,0], dtype=ScalarType)
boundary_facets = mesh.locate_entities_boundary(domain, domain.topology.dim - 1, lambda x: np.isclose(x[0], 0))
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, domain.topology.dim - 1, boundary_facets), V)

problem = fem.petsc.NonlinearProblem(F, u, bcs=[bc], graph = graph_)
solver = nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem, graph = graph_)
solver.solve(u, graph = graph_)

J_form = ufl.inner(u, u) * ufl.dx
J = fem.assemble_scalar(fem.form(J_form, graph = graph_), graph = graph_)

if __name__ == "__main__":
    dJdlambda = graph_.backprop(id(J), id(lambda_))
    dJdmu = graph_.backprop(id(J), id(mu))

    print("J = ", J)
    print("dJdλ = ", dJdlambda)
    print("dJdmu = ", dJdmu)

import unittest
class TestLinearElasticity(unittest.TestCase):
    def test_material_param_lambda(self):
        """In this test we compare the automatically constructed adjoint equations and its solution with 
        the analytically derived adjoint equations and their discretized solution.

        We compute the derivative of J with respect to the material parameter λ.
            dJ/dλ = ∂J/∂u * du/dλ                                             (1.1)
        
        We can obtain an expression for du/dλ by using the resiudal equation 0 = F(u(λ); λ)
            0 = dF/dλ  = dF/du * du/dλ + ∂F/∂λ
        
            => du/dλ = -dF/du^{-1} * ∂F/∂λ                                    (1.2)

        Inserting (1.2) into (1.1) yields
            dJ/dλ = ∂J/∂u * -dF/du^{-1} * ∂F/∂λ

        This leads to the adjoint equation
            (∂Fᵀ/∂u) * theta = - ∂Jᵀ/∂u
        and with the adjoint solution theta, we can compute the derivative of J with respect to the material parameter λ
            dJ/dλ =  thetaᵀ * ∂F/∂λ
        """

        dJdu = ufl.derivative(J_form, u)
        dFdu = ufl.derivative(F, u)

        # In order to use ufl.derivative for the derivative of F with respect to λ,
        # we need to define a new form
        DG0 = fem.FunctionSpace(domain, ("DG", 0))
        lambda_func = fem.Function(DG0, name = "lambda")
        lambda_func.vector.array[:] = ScalarType(1.25)
        F_replace = ufl.replace(F, {lambda_: lambda_func})
        dFdlambda = ufl.derivative(F_replace, lambda_func)

        dJdu = fem.assemble_vector(fem.form(dJdu)).array
        dFdlambda = fem.assemble_vector(fem.form(dFdlambda)).array
        dFdu = fem.assemble_matrix(fem.form(dFdu), bcs = [bc]).to_dense()

        adjoint_solution = np.linalg.solve(dFdu.transpose(), -dJdu.transpose())
        dJdlambda = adjoint_solution.transpose() @ dFdlambda

        self.assertTrue(np.allclose(dJdlambda, graph_.backprop(id(J), id(lambda_))))

    def test_material_param_mu(self):
        """In this test we compare the automatically constructed adjoint equations and its solution with 
        the analytically derived adjoint equations and their discretized solution.

        We compute the derivative of J with respect to the material parameter mu.
            dJ/dmu = ∂J/∂u * du/dmu                                             (1.1)
        
        We can obtain an expression for du/dλ by using the resiudal equation 0 = F(u(mu); mu)
            0 = dF/dmu  = dF/du * du/dmu + ∂F/∂mu
        
            => du/dmu = -dF/du^{-1} * ∂F/∂mu                                    (1.2)

        Inserting (1.2) into (1.1) yields
            dJ/dmu = ∂J/∂u * -dF/du^{-1} * ∂F/∂mu

        This leads to the adjoint equation
            (∂Fᵀ/∂u) * theta = - ∂Jᵀ/∂u
        and with the adjoint solution theta, we can compute the derivative of J with respect to the material parameter mu
            dJ/dmu =  thetaᵀ * ∂F/∂mu
        """

        dJdu = ufl.derivative(J_form, u)
        dFdu = ufl.derivative(F, u)

        # In order to use ufl.derivative for the derivative of F with respect to mu,
        # we need to define a new form
        DG0 = fem.FunctionSpace(domain, ("DG", 0))
        mu_func = fem.Function(DG0, name = "mu")
        mu_func.vector.array[:] = ScalarType(1.0)
        F_replace = ufl.replace(F, {mu: mu_func})
        dFdmu = ufl.derivative(F_replace, mu_func)

        dJdu = fem.assemble_vector(fem.form(dJdu)).array
        dFdmu = fem.assemble_vector(fem.form(dFdmu)).array
        dFdu = fem.assemble_matrix(fem.form(dFdu), bcs = [bc]).to_dense()

        adjoint_solution = np.linalg.solve(dFdu.transpose(), -dJdu.transpose())
        dJdmu = adjoint_solution.transpose() @ dFdmu

        self.assertTrue(np.allclose(dJdmu, graph_.backprop(id(J), id(mu))))