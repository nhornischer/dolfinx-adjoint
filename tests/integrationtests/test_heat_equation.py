"""
Integration tests for the Heat Equation adjoint gradients.

This test module verifies that the automatic differentiation implementation
correctly computes gradients using the adjoint method for time-dependent problems.
"""

import numpy as np
import pytest
import ufl
from dolfinx import fem, mesh, nls
from mpi4py import MPI
from petsc4py.PETSc import ScalarType

from dolfinx_adjoint import *


@pytest.fixture(scope="module")
def heat_equation_problem():
    """Set up the heat equation problem that will be used in all tests."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 32, 32, mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", 1))

    dt = 0.01
    T = 0.05

    # Create true data set
    true_initial = fem.Function(V, name="u_true_initial")
    true_initial.interpolate(
        lambda x: np.sin(2 * np.pi * x[0]) * np.sin(2 * np.pi * x[1])
    )
    u_prev = true_initial.copy()
    u_next = true_initial.copy()

    v = ufl.TestFunction(V)
    dt_constant = fem.Constant(domain, ScalarType(dt))

    # Set dirichlet boundary conditions
    uD = fem.Function(V)
    uD.interpolate(lambda x: 0.0 + 0.0 * x[0])
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)

    F = (
        ufl.inner((u_next - u_prev) / dt_constant, v) * ufl.dx
        + ufl.inner(ufl.grad(u_next), ufl.grad(v)) * ufl.dx
    )
    problem = fem.petsc.NewtonSolverNonlinearProblem(
        F, u_next, petsc_options_prefix="nls_"
    )
    solver = nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)

    t = 0.0
    while t < T:
        solver.solve(u_next)
        u_prev.vector[:] = u_next.vector[:]
        t += dt
    true_data = u_next.copy()

    # Create test data
    graph_ = Graph()

    # Set the initial values of the temperature variable u
    initial_guess = fem.Function(V, name="initial_guess", graph=graph_)
    initial_guess.interpolate(
        lambda x: 15.0 * x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1])
    )
    u_prev = initial_guess.copy(graph=graph_, name="u_prev")
    u_next = initial_guess.copy(graph=graph_, name="u_next")

    F = (
        ufl.inner((u_next - u_prev) / dt_constant, v) * ufl.dx
        + ufl.inner(ufl.grad(u_next), ufl.grad(v)) * ufl.dx
    )
    t = 0.0
    i = 0

    # Store the states for the whole time-domain
    u_iterations = [u_next.copy()]
    while t < T:
        i += 1
        F = (
            ufl.inner((u_next - u_prev) / dt_constant, v) * ufl.dx
            + ufl.inner(ufl.grad(u_next), ufl.grad(v)) * ufl.dx
        )
        problem = fem.petsc.NewtonSolverNonlinearProblem(
            F, u_next, graph=graph_, petsc_options_prefix="nls_"
        )
        solver = nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem, graph=graph_)

        solver.solve(u_next, graph=graph_, version=i)
        t += dt
        u_prev.assign(u_next, graph=graph_, version=i)

        # Storing the iterations for later visualization and testing
        u_iterations.append(u_next.copy())

    alpha = fem.Constant(domain, ScalarType(1.0e-6))

    J_form = (
        ufl.inner(true_data - u_next, true_data - u_next) * ufl.dx
        + alpha * ufl.inner(ufl.grad(initial_guess), ufl.grad(initial_guess)) * ufl.dx
    )
    J = fem.assemble_scalar(fem.form(J_form, graph=graph_), graph=graph_)

    return {
        "graph_": graph_,
        "domain": domain,
        "V": V,
        "J_form": J_form,
        "J": J,
        "initial_guess": initial_guess,
        "u_next": u_next,
        "u_prev": u_prev,
        "F": F,
        "u_iterations": u_iterations,
        "dt_constant": dt_constant,
        "v": v,
    }


def test_Heat_initial(heat_equation_problem):
    """
    Test gradient of J with respect to the initial condition.

    We compute the derivative of J with respect to the initial guess $u_0$:

        $$\\frac{dJ}{du_0} = \\frac{\\partial J}{\\partial u_N} \\frac{du_N}{du_{N-1}} \\cdots \\frac{du_1}{du_0} + \\frac{\\partial J}{\\partial u_0}$$

    where $u_i$ is the solution at time step $i$.

    The first and last terms $\\frac{\\partial J}{\\partial u_N}$ and $\\frac{\\partial J}{\\partial u_0}$
    can be easily obtained using UFL.

    The term $\\frac{du_N}{du_{N-1}}$ is obtained by solving the adjoint equation. Taking the derivative
    of the residual equation $F(u_N, u_{N-1}) = 0$ with respect to $u_{N-1}$:

        $$\\frac{dF}{du_{N-1}} = \\frac{\\partial F}{\\partial u_N} \\frac{du_N}{du_{N-1}} + \\frac{\\partial F}{\\partial u_{N-1}} = 0$$

        $$\\Rightarrow \\frac{du_N}{du_{N-1}} = - \\left(\\frac{\\partial F}{\\partial u_N}\\right)^{-1} \\frac{\\partial F}{\\partial u_{N-1}}$$

    This leads to the first adjoint equation:

        $$\\left(\\frac{\\partial F^T}{\\partial u_N}\\right) \\lambda_N = - \\frac{\\partial J^T}{\\partial u_N}$$

    and subsequent adjoint equations:

        $$\\left(\\frac{\\partial F^T}{\\partial u_{i-1}}\\right) \\lambda_{i-1} = - \\lambda_i \\frac{\\partial F^T}{\\partial u_{i-1}}$$

    Finally:

        $$\\frac{dJ}{du_0} = \\lambda_1^T \\frac{\\partial F}{\\partial u_0} + \\frac{\\partial J}{\\partial u_0}$$
    """
    graph_ = heat_equation_problem["graph_"]
    J_form = heat_equation_problem["J_form"]
    J = heat_equation_problem["J"]
    initial_guess = heat_equation_problem["initial_guess"]
    u_next = heat_equation_problem["u_next"]
    u_prev = heat_equation_problem["u_prev"]
    F = heat_equation_problem["F"]
    u_iterations = heat_equation_problem["u_iterations"]

    dJdu = ufl.derivative(J_form, u_next)
    dJdu_0 = ufl.derivative(J_form, initial_guess)

    dJdu_0 = fem.assemble_vector(fem.form(dJdu_0)).array
    dJdu = fem.assemble_vector(fem.form(dJdu)).array

    rhs = dJdu

    for i in range(len(u_iterations) - 1, 0, -1):
        F_i = ufl.replace(F, {u_next: u_iterations[i], u_prev: u_iterations[i - 1]})
        dF_idu_i = ufl.derivative(F_i, u_iterations[i])
        dF_idu_i = fem.assemble_matrix(fem.form(dF_idu_i)).to_dense()

        lambda_i = np.linalg.solve(dF_idu_i.transpose(), -rhs.transpose())

        dF_idu_i_1 = ufl.derivative(F_i, u_iterations[i - 1])
        dF_idu_i_1 = fem.assemble_matrix(fem.form(dF_idu_i_1)).to_dense()
        rhs = lambda_i.transpose() @ dF_idu_i_1

    gradient = rhs + dJdu_0

    # Compare automatic differentiation result with explicit adjoint calculation
    assert np.allclose(graph_.backprop(id(J), id(initial_guess)), gradient)
