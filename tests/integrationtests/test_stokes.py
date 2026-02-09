"""
Integration tests for the Stokes equation adjoint gradients.

This test module verifies that the automatic differentiation implementation
correctly computes gradients using the adjoint method by comparing against
explicit adjoint calculations.
"""

import gmsh
import numpy as np
import pytest
import ufl
from dolfinx import fem, io, nls
from mpi4py import MPI
from petsc4py.PETSc import ScalarType

from dolfinx_adjoint import *


@pytest.fixture(scope="module")
def stokes_problem():
    """Set up the Stokes problem that will be used in all tests."""
    # Mesh parameters
    gmsh.initialize()
    L = 2.2
    H = 0.41
    c_x = 0.2
    c_y = 0.2
    r = 0.05
    gdim = 2
    mesh_comm = MPI.COMM_WORLD
    model_rank = 0

    fluid_marker = 1
    inlet_marker, outlet_marker, wall_marker, obstacle_marker = 2, 3, 4, 5
    inflow, outflow, walls, obstacle = [], [], [], []

    # Mesh generation
    if mesh_comm.rank == model_rank:
        rectangle = gmsh.model.occ.addRectangle(0, 0, 0, L, H, tag=1)
        circle = gmsh.model.occ.addDisk(c_x, c_y, 0, r, r)
        fluid = gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, circle)])
        gmsh.model.occ.synchronize()

        volumes = gmsh.model.getEntities(dim=gdim)
        assert len(volumes) == 1
        gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker)
        gmsh.model.setPhysicalName(volumes[0][0], fluid_marker, "Fluid")

        boundaries = gmsh.model.getBoundary(volumes, oriented=False)
        for boundary in boundaries:
            center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
            if np.allclose(center_of_mass, [0, H / 2, 0]):
                inflow.append(boundary[1])
            elif np.allclose(center_of_mass, [L, H / 2, 0]):
                outflow.append(boundary[1])
            elif np.allclose(center_of_mass, [L / 2, H, 0]) or np.allclose(
                center_of_mass, [L / 2, 0, 0]
            ):
                walls.append(boundary[1])
            else:
                obstacle.append(boundary[1])
        gmsh.model.addPhysicalGroup(1, walls, wall_marker)
        gmsh.model.setPhysicalName(1, wall_marker, "Walls")
        gmsh.model.addPhysicalGroup(1, inflow, inlet_marker)
        gmsh.model.setPhysicalName(1, inlet_marker, "Inlet")
        gmsh.model.addPhysicalGroup(1, outflow, outlet_marker)
        gmsh.model.setPhysicalName(1, outlet_marker, "Outlet")
        gmsh.model.addPhysicalGroup(1, obstacle, obstacle_marker)
        gmsh.model.setPhysicalName(1, obstacle_marker, "Obstacle")

        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "EdgesList", obstacle)
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "IField", 1)
        gmsh.model.mesh.field.setNumber(2, "LcMin", 0.01)
        gmsh.model.mesh.field.setNumber(2, "LcMax", 0.04)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 0)
        gmsh.model.mesh.field.setNumber(2, "DistMax", H)
        gmsh.model.mesh.field.add("Min", 5)
        gmsh.model.mesh.field.setNumbers(5, "FieldsList", [2])
        gmsh.model.mesh.field.setAsBackgroundMesh(5)
        gmsh.model.mesh.generate(2)

    mesh, _, ft = io.gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
    ft.name = "Facet markers"

    graph_ = Graph()

    # Function spaces as Mixed Element space
    from basix.ufl import element
    u_elem = element("Lagrange", mesh.basix_cell(), 2, shape = (2,))
    p_elem = element("Lagrange", mesh.basix_cell(), 1)

    v_elem = ufl.MixedElement([u_elem, p_elem])
    V = fem.functionspace(mesh, v_elem)
    V_u, V_u_map = V.sub(0).collapse()
    V_p, V_p_map = V.sub(1).collapse()

    up = fem.Function(V, name="up", graph=graph_)
    u, p = ufl.split(up)

    vq = ufl.TestFunction(V)
    v, q = ufl.split(vq)

    # Boundary conditions
    h = fem.Function(V_u, name="f")
    speed = 0.3
    h.interpolate(
        lambda x: np.stack((speed * 4 * x[1] * (H - x[1]) / (H * H), 0.0 * x[0]))
    )
    g = fem.Function(V_u, name="g", graph=graph_)
    noslip = fem.Function(V_u, name="noslip")
    outflow = fem.Function(V_p, name="outflow")
    outflow.interpolate(lambda x: 0.0 * x[0] + 0.0)

    dofs_walls = fem.locate_dofs_topological(
        (V.sub(0), V_u), 1, ft.indices[ft.values == wall_marker]
    )
    dofs_inflow = fem.locate_dofs_topological(
        (V.sub(0), V_u), 1, ft.indices[ft.values == inlet_marker]
    )
    dofs_outflow = fem.locate_dofs_topological(
        (V.sub(1), V_p), 1, ft.indices[ft.values == outlet_marker]
    )
    dofs_obstacle = fem.locate_dofs_topological(
        (V.sub(0), V_u), 1, ft.indices[ft.values == obstacle_marker]
    )

    bc_dofs_total = np.concatenate(
        [dofs_walls[0], dofs_inflow[0], dofs_outflow[0], dofs_obstacle[0]]
    )

    bcs = [
        fem.dirichletbc(h, dofs_inflow, V.sub(0)),
        fem.dirichletbc(g, dofs_obstacle, V.sub(0), graph=graph_, map=V_u_map),
        fem.dirichletbc(noslip, dofs_walls, V.sub(0)),
        fem.dirichletbc(outflow, dofs_outflow, V.sub(1)),
    ]

    # Parameters
    nu = fem.Constant(mesh, ScalarType(1.0), name="ν", graph=graph_)
    alpha = 10.0
    f = fem.Function(V_u, name="f")
    f.interpolate(lambda x: (0.0 * x[0], 0.0 + 0.0 * x[1]))

    # Variational formulation
    a = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - ufl.div(v) * p * ufl.dx
        + q * ufl.div(u) * ufl.dx
    )
    L = ufl.inner(f, v) * ufl.dx
    F = a - L

    # Define the problem solver
    problem = fem.petsc.NewtonSolverNonlinearProblem(
        F, up, bcs=bcs, graph=graph_, petsc_options_prefix="nls_"
    )
    solver = nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem, graph=graph_)

    solver.solve(up, graph=graph_)

    # Define the objective function
    dObs = ufl.Measure(
        "ds", domain=mesh, subdomain_data=ft, subdomain_id=obstacle_marker
    )
    J_form = (
        0.5 * ufl.inner(ufl.grad(u), ufl.grad(u)) * ufl.dx
        + alpha / 2 * ufl.inner(g, g) * dObs
    )

    J = fem.assemble_scalar(fem.form(J_form, graph=graph_), graph=graph_)

    gmsh.finalize()

    return {
        "graph_": graph_,
        "mesh": mesh,
        "V": V,
        "V_u": V_u,
        "V_p": V_p,
        "V_u_map": V_u_map,
        "up": up,
        "u": u,
        "p": p,
        "g": g,
        "nu": nu,
        "F": F,
        "J_form": J_form,
        "J": J,
        "bcs": bcs,
        "bc_dofs_total": bc_dofs_total,
        "dofs_obstacle": dofs_obstacle,
        "problem": problem,
        "dObs": dObs,
    }


def test_Stokes_dJdnu(stokes_problem):
    """
    Test gradient of J with respect to viscosity ν.

    We compute the derivative of J with respect to ν:

        dJ/dν = ∂J/∂u * du/dν + ∂J/∂ν                               (2.1)

    The first term ∂J/∂u is easy to compute. The second term du/dν is handled
    using the adjoint problem:

        dF/dν = ∂F/∂u * du/dν + ∂F/∂ν = 0
        => du/dν = -(∂F/∂u)^{-1} * ∂F/∂ν                              (2.2)

    Inserting (2.2) into (2.1) yields:

        dJ/dν = - ∂J/∂u * (∂F/∂u)^{-1} * ∂F/∂ν + ∂J/∂ν                (2.3)

    This leads to the adjoint equation:

        (∂F^T/∂u) * θ = -∂J^T/∂u

    And with the adjoint solution θ:

        dJ/dν = θ^T * ∂F/∂ν + ∂J/∂ν                                  (2.5)
    """
    mesh = stokes_problem["mesh"]
    up = stokes_problem["up"]
    nu = stokes_problem["nu"]
    F = stokes_problem["F"]
    J_form = stokes_problem["J_form"]
    J = stokes_problem["J"]
    bcs = stokes_problem["bcs"]
    bc_dofs_total = stokes_problem["bc_dofs_total"]
    graph_ = stokes_problem["graph_"]

    DG0 = fem.functionspace(mesh, ("DG", 0))
    nu_function = fem.Function(DG0, name="nu")
    nu_function.vector.array[:] = ScalarType(1.0)

    J_form_replaced = ufl.replace(J_form, {nu: nu_function})
    F_replaced = ufl.replace(F, {nu: nu_function})

    dJdu = ufl.derivative(J_form, up)
    dJdnu = ufl.derivative(J_form_replaced, nu_function)
    dFdnu = ufl.derivative(F_replaced, nu_function)
    dFdu = ufl.derivative(F, up)

    dJdu = fem.assemble_vector(fem.form(dJdu)).array
    dJdnu = fem.assemble_scalar(fem.form(dJdnu))
    dFdnu = fem.assemble_vector(fem.form(dFdnu)).array
    dFdu = fem.assemble_matrix(fem.form(dFdu), bcs=bcs).to_dense()

    # Apply the boundary conditions to the rhs of the adjoint problem
    for bc_dof in bc_dofs_total:
        dJdu[int(bc_dof)] = 0

    adjoint_solution = np.linalg.solve(dFdu.transpose(), -dJdu.transpose())
    gradient = adjoint_solution.transpose() @ dFdnu + dJdnu

    assert np.allclose(graph_.backprop(id(J), id(nu)), gradient)


def test_Stokes_dJdg(stokes_problem):
    """
    Test gradient of J with respect to boundary condition g.

    We calculate the derivative of the objective function with respect to the
    Dirichlet boundary condition g:

        dJ(u,p,g)/dg = ∂J/∂up * ∂up/∂g + ∂J/∂g

    Since u is defined on a mixed element space, we need to map the derivatives
    appropriately between function spaces.

    From the residual equation 0 = F(u, p):

        0 = ∂F/∂up * ∂up/∂g + ∂F/∂g

    This leads to the adjoint equation:

        (∂F/∂up)^T * θ = -(∂J/∂up)^T

    And the gradient computation:

        dJ/dg = θ^T * ∂F/∂g + ∂J/∂g

    We extract only the boundary values by multiplying with an identity matrix
    nonzero on the boundary.
    """
    V = stokes_problem["V"]
    V_u = stokes_problem["V_u"]
    V_u_map = stokes_problem["V_u_map"]
    up = stokes_problem["up"]
    g = stokes_problem["g"]
    F = stokes_problem["F"]
    J_form = stokes_problem["J_form"]
    J = stokes_problem["J"]
    bcs = stokes_problem["bcs"]
    bc_dofs_total = stokes_problem["bc_dofs_total"]
    dofs_obstacle = stokes_problem["dofs_obstacle"]
    problem = stokes_problem["problem"]
    graph_ = stokes_problem["graph_"]

    argument = ufl.TrialFunction(V)
    dJdu = ufl.derivative(J_form, up, argument)

    dJdg = ufl.derivative(J_form, g)

    argument = ufl.TrialFunction(V)
    dFdu = ufl.derivative(F, up, argument)

    dJdu = fem.assemble_vector(fem.form(dJdu)).array
    dJdg = fem.assemble_vector(fem.form(dJdg)).array

    dFdg = fem.assemble_matrix(problem._a).to_dense()
    dFdu = fem.assemble_matrix(fem.form(dFdu), bcs=bcs).to_dense()

    # Apply the boundary conditions to the rhs of the adjoint problem
    for bc_dof in bc_dofs_total:
        dJdu[int(bc_dof)] = 0

    adjoint_solution = np.linalg.solve(dFdu.transpose(), -dJdu.transpose())

    gradient = adjoint_solution.transpose() @ dFdg

    matrix = np.zeros((len(gradient), len(gradient)))
    for index in dofs_obstacle[0]:
        matrix[index, index] = 1.0

    gradient = (matrix @ gradient)[V_u_map] + dJdg
    assert np.allclose(gradient, graph_.backprop(id(J), id(g)))
