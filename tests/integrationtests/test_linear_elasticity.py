"""
Integration tests for the Linear Elasticity adjoint gradients.

This test module verifies that the automatic differentiation implementation
correctly computes gradients using the adjoint method by comparing against
explicit adjoint calculations.
"""

import numpy as np
import pytest
import ufl
from basix.ufl import element
from dolfinx import fem, mesh, nls
from mpi4py import MPI
from petsc4py.PETSc import ScalarType

from dolfinx_adjoint import *


@pytest.fixture(scope="module")
def linear_elasticity_problem():
    """Set up the linear elasticity problem that will be used in all tests."""
    # Scaled variable
    L = 1
    W = 0.1
    rho = 1
    delta = W / L
    gamma = 0.4 * delta**2
    g = gamma

    graph_ = Graph()

    domain = mesh.create_box(
        MPI.COMM_WORLD,
        [np.array([0, 0, 0]), np.array([L, W, W])],
        [30, 10, 10],
        cell_type=mesh.CellType.hexahedron,
    )

    vector_element = element("Lagrange", domain.basix_cell(), 1, shape=(3,))
    V = fem.functionspace(domain, vector_element)
    ds = ufl.Measure("ds", domain=domain)

    u = fem.Function(V, name="Deformation", graph=graph_)
    v = ufl.TestFunction(V)

    f = fem.Constant(domain, ScalarType((0, 0, -rho * g)))
    T = fem.Constant(domain, ScalarType((0, 0, 0)))
    lambda_ = fem.Constant(domain, ScalarType(1.0), graph=graph_)
    mu = fem.Constant(domain, ScalarType(1.25), graph=graph_)

    a = (
        ufl.inner(
            lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u))
            + 2 * mu * ufl.sym(ufl.grad(u)),
            ufl.sym(ufl.grad(v)),
        )
        * ufl.dx
    )
    L = ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * ds
    F = a - L

    # Boundary condition describing the clamped left side of the beam
    u_D = np.array([0, 0, 0], dtype=ScalarType)
    boundary_facets = mesh.locate_entities_boundary(
        domain, domain.topology.dim - 1, lambda x: np.isclose(x[0], 0)
    )
    bc = fem.dirichletbc(
        u_D, fem.locate_dofs_topological(V, domain.topology.dim - 1, boundary_facets), V
    )

    problem = fem.petsc.NonlinearProblem(F, u, bcs=[bc], graph=graph_)
    solver = nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem, graph=graph_)
    solver.solve(u, graph=graph_)

    J_form = ufl.inner(u, u) * ufl.dx
    J = fem.assemble_scalar(fem.form(J_form, graph=graph_), graph=graph_)

    return {
        "graph_": graph_,
        "domain": domain,
        "V": V,
        "u": u,
        "lambda_": lambda_,
        "mu": mu,
        "F": F,
        "J_form": J_form,
        "J": J,
        "bc": bc,
    }


def test_material_param_lambda(linear_elasticity_problem):
    """
    Test gradient of J with respect to the material parameter λ.

    We compute the derivative of J with respect to the material parameter λ:

        dJ/dλ = ∂J/∂u * du/dλ                                             (1.1)

    We can obtain an expression for du/dλ by using the residual equation 0 = F(u(λ); λ):

        0 = dF/dλ = dF/du * du/dλ + ∂F/∂λ
        => du/dλ = -dF/du^{-1} * ∂F/∂λ                                    (1.2)

    Inserting (1.2) into (1.1) yields:

        dJ/dλ = ∂J/∂u * -dF/du^{-1} * ∂F/∂λ

    This leads to the adjoint equation:

        (∂F^T/∂u) * θ = -∂J^T/∂u

    And with the adjoint solution θ, we can compute the derivative of J with respect to λ:

        dJ/dλ = θ^T * ∂F/∂λ
    """
    domain = linear_elasticity_problem["domain"]
    u = linear_elasticity_problem["u"]
    lambda_ = linear_elasticity_problem["lambda_"]
    F = linear_elasticity_problem["F"]
    J_form = linear_elasticity_problem["J_form"]
    J = linear_elasticity_problem["J"]
    bc = linear_elasticity_problem["bc"]
    graph_ = linear_elasticity_problem["graph_"]

    dJdu = ufl.derivative(J_form, u)
    dFdu = ufl.derivative(F, u)

    # Define a new form to use ufl.derivative for the derivative of F with respect to λ
    DG0 = fem.functionspace(domain, ("DG", 0))
    lambda_func = fem.Function(DG0, name="lambda")
    lambda_func.vector.array[:] = ScalarType(1.0)
    F_replace = ufl.replace(F, {lambda_: lambda_func})
    dFdlambda = ufl.derivative(F_replace, lambda_func)

    dJdu = fem.assemble_vector(fem.form(dJdu)).array
    dFdlambda = fem.assemble_vector(fem.form(dFdlambda)).array
    dFdu = fem.assemble_matrix(fem.form(dFdu), bcs=[bc]).to_dense()

    adjoint_solution = np.linalg.solve(dFdu.transpose(), -dJdu.transpose())
    dJdlambda = adjoint_solution.transpose() @ dFdlambda

    # Compare automatic differentiation result with explicit adjoint calculation
    assert np.allclose(dJdlambda, graph_.backprop(id(J), id(lambda_)))


def test_material_param_mu(linear_elasticity_problem):
    """
    Test gradient of J with respect to the material parameter μ.

    We compute the derivative of J with respect to the material parameter μ:

        dJ/dμ = ∂J/∂u * du/dμ                                             (1.1)

    We can obtain an expression for du/dμ by using the residual equation 0 = F(u(μ); μ):

        0 = dF/dμ = dF/du * du/dμ + ∂F/∂μ
        => du/dμ = -dF/du^{-1} * ∂F/∂μ                                    (1.2)

    Inserting (1.2) into (1.1) yields:

        dJ/dμ = ∂J/∂u * -dF/du^{-1} * ∂F/∂μ

    This leads to the adjoint equation:

        (∂F^T/∂u) * θ = -∂J^T/∂u

    And with the adjoint solution θ, we can compute the derivative of J with respect to μ:

        dJ/dμ = θ^T * ∂F/∂μ
    """
    domain = linear_elasticity_problem["domain"]
    u = linear_elasticity_problem["u"]
    mu = linear_elasticity_problem["mu"]
    F = linear_elasticity_problem["F"]
    J_form = linear_elasticity_problem["J_form"]
    J = linear_elasticity_problem["J"]
    bc = linear_elasticity_problem["bc"]
    graph_ = linear_elasticity_problem["graph_"]

    dJdu = ufl.derivative(J_form, u)
    dFdu = ufl.derivative(F, u)

    # Define a new form to use ufl.derivative for the derivative of F with respect to μ
    DG0 = fem.functionspace(domain, ("DG", 0))
    mu_func = fem.Function(DG0, name="mu")
    mu_func.vector.array[:] = ScalarType(1.25)
    F_replace = ufl.replace(F, {mu: mu_func})
    dFdmu = ufl.derivative(F_replace, mu_func)

    dJdu = fem.assemble_vector(fem.form(dJdu)).array
    dFdmu = fem.assemble_vector(fem.form(dFdmu)).array
    dFdu = fem.assemble_matrix(fem.form(dFdu), bcs=[bc]).to_dense()

    adjoint_solution = np.linalg.solve(dFdu.transpose(), -dJdu.transpose())
    dJdmu = adjoint_solution.transpose() @ dFdmu

    # Compare automatic differentiation result with explicit adjoint calculation
    assert np.allclose(dJdmu, graph_.backprop(id(J), id(mu)))
