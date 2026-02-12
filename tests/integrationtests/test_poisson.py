"""
Integration tests for the Poisson equation adjoint gradients.

This test module verifies that the automatic differentiation implementation
correctly computes gradients using the adjoint method by comparing against
explicit adjoint calculations.
"""

import dolfinx
import numpy as np
import pytest
import ufl
from dolfinx import fem, mesh, nls
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py.PETSc import ScalarType

from dolfinx_adjoint import *


@pytest.fixture(
    scope="module",
    params=[mesh.CellType.triangle, mesh.CellType.quadrilateral],
    ids=["triangle", "quadrilateral"],
)
def poisson_problem(request):
    """Set up the Poisson problem that will be used in all tests."""
    # Create graph object to store the computational graph
    graph_ = Graph()

    # Define mesh and finite element space
    cell_type = request.param
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


def _convergence_rates(errors: np.ndarray, steps: np.ndarray) -> list:
    rates = []
    for i in range(1, len(steps)):
        rates.append(
            np.log(errors[i] / errors[i - 1]) / np.log(steps[i] / steps[i - 1])
        )
    return rates


def test_Poisson_graph_recalculate(poisson_problem):
    """Ensure graph.recalculate updates J after modifying f."""
    graph_ = poisson_problem["graph_"]
    f = poisson_problem["f"]
    J = poisson_problem["J"]

    J_node = graph_.get_node(id(J))
    assert J_node is not None

    comm = f.function_space.mesh.comm
    J0 = comm.allreduce(J_node.object, op=MPI.SUM)

    f_org = f.x.array.copy()

    f.x.array[:] = f_org + 0.5
    graph_.recalculate()
    J1 = comm.allreduce(J_node.object, op=MPI.SUM)
    assert not np.isclose(J1, J0)

    f.x.array[:] = f_org
    graph_.recalculate()
    J2 = comm.allreduce(J_node.object, op=MPI.SUM)
    np.testing.assert_allclose(J2, J0)


def test_Poisson_taylor_f(poisson_problem):
    """Taylor test for J with respect to the forcing term f (graph backprop)."""
    graph_ = poisson_problem["graph_"]
    f = poisson_problem["f"]
    J = poisson_problem["J"]

    J_node = graph_.get_node(id(J))
    assert J_node is not None

    comm = f.function_space.mesh.comm
    J0 = comm.allreduce(J_node.object, op=MPI.SUM)

    df = fem.Function(f.function_space)
    df.interpolate(lambda x: x[0] + np.sin(x[1]))

    grad = graph_.backprop(id(J), id(f))
    grad_array = grad.array if hasattr(grad, "array") else np.array(grad)
    dJ = comm.allreduce(np.dot(grad_array, df.x.array), op=MPI.SUM)

    f_org = f.x.array.copy()
    step_length = 1e-2
    steps = [step_length * (0.5**i) for i in range(4)]
    errors0 = []
    errors = []
    try:
        for h in steps:
            f.x.array[:] = f_org + h * df.x.array
            graph_.recalculate()
            Jh = comm.allreduce(J_node.object, op=MPI.SUM)
            errors0.append(abs(Jh - J0))
            errors.append(abs(Jh - J0 - h * dJ))
    finally:
        f.x.array[:] = f_org
        graph_.recalculate()

    rates0 = _convergence_rates(errors0, steps)
    np.testing.assert_allclose(rates0, 1.0, atol=0.2)
    rates = _convergence_rates(errors, steps)
    np.testing.assert_allclose(rates, 2.0, atol=0.2)


def test_Poisson_taylor_nu(poisson_problem):
    """Taylor test for J with respect to the diffusion coefficient nu."""
    graph_ = poisson_problem["graph_"]
    nu = poisson_problem["nu"]
    J = poisson_problem["J"]

    J_node = graph_.get_node(id(J))
    assert J_node is not None

    comm = nu.domain.comm
    J0 = comm.allreduce(J_node.object, op=MPI.SUM)

    direction = 1.0
    grad = graph_.backprop(id(J), id(nu))
    dJ = comm.allreduce(float(grad) * direction, op=MPI.SUM)

    nu_org = float(nu.value)
    step_length = 1e-2
    steps = [step_length * (0.5**i) for i in range(4)]
    errors0 = []
    errors = []
    try:
        for h in steps:
            nu.value = float(nu_org + h * direction)
            graph_.recalculate()
            Jh = comm.allreduce(J_node.object, op=MPI.SUM)
            errors0.append(abs(Jh - J0))
            errors.append(abs(Jh - J0 - h * dJ))
    finally:
        nu.value = nu_org
        graph_.recalculate()

    rates0 = _convergence_rates(errors0, steps)
    np.testing.assert_allclose(rates0, 1.0, atol=0.2)
    rates = _convergence_rates(errors, steps)
    np.testing.assert_allclose(rates, 2.0, atol=0.2)


def test_Poisson_dJdf(poisson_problem):
    """
    Test gradient of J with respect to forcing term f.

    We turn to the explicit adjoint approach to calculate the gradient of J(u) with respect to f.
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
    F = poisson_problem["F"]
    uh = poisson_problem["uh"]
    f = poisson_problem["f"]
    J_form = poisson_problem["J_form"]
    bcs_adjoint = poisson_problem["bcs_adjoint"]
    graph_ = poisson_problem["graph_"]
    J = poisson_problem["J"]

    dFdu = ufl.derivative(F, uh)
    dFdf = ufl.derivative(F, f)
    dJdu = ufl.derivative(J_form, uh)
    dJdf = ufl.derivative(J_form, f)

    adjoint_solution = LinearProblem(
        ufl.adjoint(dFdu), -dJdu, bcs=bcs_adjoint, petsc_options_prefix="adjoint_"
    ).solve()
    gradient = ufl.action(ufl.adjoint(dFdf), adjoint_solution) + dJdf

    gradient_df = dolfinx.fem.assemble_vector(dolfinx.fem.form(gradient))
    gradient_df.scatter_reverse(dolfinx.la.InsertMode.add)
    gradient_df.scatter_forward()

    # Compare automatic differentiation result with explicit adjoint calculation
    assert np.allclose(graph_.backprop(id(J), id(f)), gradient_df.array[:])


def test_Poisson_dJdnu(poisson_problem):
    """
    Test gradient of J with respect to diffusion coefficient ν.

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
    domain = poisson_problem["domain"]
    F = poisson_problem["F"]
    uh = poisson_problem["uh"]
    nu = poisson_problem["nu"]
    J_form = poisson_problem["J_form"]
    bcs_adjoint = poisson_problem["bcs_adjoint"]
    graph_ = poisson_problem["graph_"]
    J = poisson_problem["J"]

    DG0 = fem.functionspace(domain, ("DG", 0))
    nu_function = fem.Function(DG0, name="nu")
    nu_function.x.array[:] = ScalarType(1.0)

    J_form_replaced = ufl.replace(J_form, {nu: nu_function})
    F_replaced = ufl.replace(F, {nu: nu_function})

    dJdu = ufl.derivative(J_form, uh)
    dJdnu = ufl.derivative(J_form_replaced, nu_function)
    dFdnu = ufl.derivative(F_replaced, nu_function)
    dFdu = ufl.derivative(F, uh)

    adjoint_solution = LinearProblem(
        ufl.adjoint(dFdu), -dJdu, bcs=bcs_adjoint, petsc_options_prefix="adjoint_"
    ).solve()
    gradient = ufl.action(ufl.adjoint(dFdnu), adjoint_solution) + dJdnu
    gradient = dolfinx.fem.assemble_scalar(dolfinx.fem.form(gradient))

    # Compare automatic differentiation result with explicit adjoint calculation
    assert np.allclose(graph_.backprop(id(J), id(nu)), gradient)


def test_Poisson_dJdbc(poisson_problem):
    """
    Test gradient of J with respect to boundary condition u_D.

    We turn to the explicit adjoint approach to calculate the gradient of J(u) with respect to u_D on the left boundary.
    Therefore, we first define a new reduced functional R(u_D) = J(u(u_D),u_D) and take the derivative with respect to u_D:
        dJ/du_D = ∂J/∂u * du/du_D + ∂J/∂u_D                         (3.1)

    The first term ∂J/∂u is easy to compute and for the second term we use the adjoint problem similar to (1.2)
        dF/du_D = ∂F/∂u * du/du_D + ∂F/∂u_D = 0
        => du/du_D = -(∂F/∂u)^-1 * ∂F/∂u_D                          (3.2)

    Inserting (3.2) into (3.1) yields
        dJ/du_D = - ∂J/∂u * (∂F/∂u)^-1 * ∂F/∂u_D + ∂J/∂u_D          (3.3)

    We first need to look closer to the term ∂F/∂u_D.

    Since the boundary condition is not directly in the ufl representation of the form F,
    we need to look at how the boundary conditions are applied when constructing a nonlinear problem.
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
    F = poisson_problem["F"]
    uh = poisson_problem["uh"]
    uD_L = poisson_problem["uD_L"]
    J_form = poisson_problem["J_form"]
    boundary_dofs_L = poisson_problem["boundary_dofs_L"]
    bcs_adjoint = poisson_problem["bcs_adjoint"]
    graph_ = poisson_problem["graph_"]
    J = poisson_problem["J"]

    dFdu = ufl.derivative(F, uh)
    dJdu = ufl.derivative(J_form, uh)
    dFdbc = ufl.derivative(F, uh, ufl.TrialFunction(uh.function_space))

    adjoint_solution = LinearProblem(
        ufl.adjoint(dFdu), -dJdu, bcs=bcs_adjoint, petsc_options_prefix="adjoint_"
    ).solve()
    gradient = ufl.action(ufl.adjoint(dFdbc), adjoint_solution)

    gradient = dolfinx.fem.assemble_vector(dolfinx.fem.form(gradient))
    gradient.scatter_reverse(dolfinx.la.InsertMode.add)
    gradient.scatter_forward()

    # Extract gradient values only at the boundary
    matrix = np.zeros((len(gradient.array), len(gradient.array)))
    for index in boundary_dofs_L:
        matrix[index, index] = 1.0

    gradient = matrix @ gradient.array

    # Compare automatic differentiation result with explicit adjoint calculation
    assert np.allclose(graph_.backprop(id(J), id(uD_L)), gradient)
