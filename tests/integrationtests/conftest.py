"""
Integration tests for the Poisson equation adjoint gradients.

This test module verifies that the automatic differentiation implementation
correctly computes gradients using the adjoint method by comparing against
explicit adjoint calculations.
"""

import numpy as np
import pytest
import ufl
from dolfinx import mesh
from mpi4py import MPI
from petsc4py.PETSc import ScalarType

from dolfinx_adjoint import Graph, fem, nls


@pytest.fixture(
    scope="module",
    params=[mesh.CellType.triangle, mesh.CellType.quadrilateral],
    ids=["triangle", "quadrilateral"],
)
def cell_type(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=["nonlinear", "nonlinear_newton"],
)
def solver(request):
    return request.param


@pytest.fixture(scope="module")
def poisson_problem(cell_type, solver: bool):
    """Set up the Poisson problem that will be used in all tests."""
    # Create graph object to store the computational graph
    graph_ = Graph()

    print(f"Testing with cell type: {cell_type}")
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 64, 64, cell_type)
    V = fem.functionspace(domain, ("Lagrange", 1))
    W = fem.functionspace(domain, ("DG", 0))

    # Define the basis functions and parameters
    uh = fem.Function(V, name="uₕ", graph=graph_)
    v = ufl.TestFunction(V)
    f = fem.Function(W, name="f", graph=graph_)
    nu = fem.Constant(domain, ScalarType(1.0), name="ν", graph=graph_)

    f.interpolate(lambda x: x[0] + x[1])

    # Define the variational form and the residual equation
    a = nu * ufl.inner(ufl.grad(uh), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx
    F = a - L

    # Define the boundary and the boundary conditions
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)

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

    bcs = [
        fem.dirichletbc(uD_L, boundary_dofs_L, graph=graph_),
        fem.dirichletbc(uD_R, boundary_dofs_R),
        fem.dirichletbc(uD_T, boundary_dofs_T),
        fem.dirichletbc(uD_B, boundary_dofs_B),
    ]

    # Boundary conditions of adjoint must be set to zero since there cannot be any contribution from the boundary to the gradient of J with respect to variables except for the boundary condition itself.
    bcs_adjoint = fem.dirichletbc(
        ScalarType(0.0),
        np.concatenate(
            [boundary_dofs_L, boundary_dofs_R, boundary_dofs_T, boundary_dofs_B]
        ),
        V,
    )

    # Define the problem solver and solve it
    if solver == "nonlinear":
        problem = fem.petsc.NonlinearProblem(
            F,
            uh,
            bcs=bcs,
            petsc_options_prefix="forward",
            graph=graph_,
        )
        problem.solve(graph=graph_)
    elif solver == "nonlinear_newton":
        problem = fem.petsc.NewtonSolverNonlinearProblem(
            F,
            uh,
            bcs=bcs,
            graph=graph_,
        )
        solver = nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem, graph=graph_)
        solver.solve(uh, graph=graph_)

    # Define profile g
    x = ufl.SpatialCoordinate(domain)
    g = (1 / (2 * np.pi**2)) * ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])

    # Define the objective function
    alpha = fem.Constant(domain, ScalarType(1e-6), name="α")
    J_form = 0.5 * ufl.inner(uh - g, uh - g) * ufl.dx + alpha * ufl.inner(f, f) * ufl.dx
    J = fem.assemble_scalar(fem.form(J_form, graph=graph_), graph=graph_)

    # Return all components as a dictionary
    return {
        "graph_": graph_,
        "domain": domain,
        "uh": uh,
        "f": f,
        "nu": nu,
        "F": F,
        "uD_L": uD_L,
        "boundary_dofs_L": boundary_dofs_L,
        "J_form": J_form,
        "J": J,
        "bcs_adjoint": [bcs_adjoint],
    }
