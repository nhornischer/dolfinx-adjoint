"""
In this demo we consider the Stokes equation
    - ν ∆u + ∇p = f in Ω            
          ∇ · u = 0 in Ω
with Dirichlet boundary conditions
            u = g on circle boundary
            u = h on inlet boundary
            u = 0 on wall boundaries
            p = 0 on outlet boundary

u : Ω → R² is the unknown velocity field
p : Ω → R is the unknwon pressure field
ν : Ω → R is the viscosity field
f : Ω → R² is the body force field
h : Ω → R² is the inlet velocity boudary condition
g : Ω → R² is the circle velocity boundary condition

The domain Ω is defined as a rectangle with a circle cut out of it, similar to the DFG 2D-3 benchmark of Flow past a cylinder
The inlet boundary is the left boundary, the outlet boundary is the right boundary,
and the wall boundaries are the top and bottom boundaries.

We define the objective function (or quantity of interest)
    J(u) = ½∫_Ω ∇u · ∇u dx + ½α∫_∂Ω_Circle g² ds
with the regularization parameter α = 10.

The weak formulation of the problem reads, find (u, p) ∈ V × Q such that
    a((u, p), (v, q)) = L((v, q)) ∀ (v, q) ∈ V × Q
where
    a((u, p), (v, q)) = ∫_Ω ν ∇u · ∇v dx - ∫_Ω ∇ · v p dx - ∫_Ω ∇ · u q dx 
              L(v, q) = ∫_Ω f · v dx 

We solve this problem with a residual equation
    F((u, p)) = a((u, p), (v, q)) - L((v, q)) != 0
"""

import numpy as np
import scipy.sparse as sps
import gmsh

from mpi4py import MPI
import dolfinx
from dolfinx import fem, io, nls
from dolfinx_adjoint import *
import ufl
from petsc4py.PETSc import ScalarType

import os
dir = os.path.dirname(__file__)


# We first need to create a graph object to store the computational graph. 
# This is done explicitly to maintain the guideline of FEniCSx.
# Every function that is created with a graph object will be added to the graph
# and its gradient will be computed automatically. 

graph_ = graph.Graph()

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
    rectangle = gmsh.model.occ.addRectangle(0,0,0, L, H, tag=1)
    circle = gmsh.model.occ.addDisk(c_x, c_y, 0, r, r)
    fluid = gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, circle)])
    gmsh.model.occ.synchronize()

    volumes = gmsh.model.getEntities(dim=gdim)
    assert(len(volumes) == 1)
    gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker)
    gmsh.model.setPhysicalName(volumes[0][0], fluid_marker, "Fluid")

    boundaries = gmsh.model.getBoundary(volumes, oriented=False)
    for boundary in boundaries:
        center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
        if np.allclose(center_of_mass, [0, H/2, 0]):
            inflow.append(boundary[1])
        elif np.allclose(center_of_mass, [L, H/2, 0]):
            outflow.append(boundary[1])
        elif np.allclose(center_of_mass, [L/2, H, 0]) or np.allclose(center_of_mass, [L/2, 0, 0]):
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

# Function spaces as Mixed Element space
u_elem = ufl.VectorElement("CG", mesh.ufl_cell(), 2)
p_elem = ufl.FiniteElement("CG", mesh.ufl_cell(), 1)

v_elem = ufl.MixedElement([u_elem, p_elem])
V = fem.FunctionSpace(mesh, v_elem)
V_u, V_u_map = V.sub(0).collapse()
V_p, V_p_map = V.sub(1).collapse()

up = fem.Function(V, name="up", graph=graph_)
(u, p) = ufl.split(up)

vq = ufl.TestFunction(V)
(v, q) = ufl.split(vq)

# Boundary conditions   
h = fem.Function(V_u, name="f")             # Inflow Dirichlet boundary condition
speed = 0.3
h.interpolate(lambda x: np.stack((speed*4*x[1]*(H-x[1])/(H*H), 0.0* x[0])))
g = fem.Function(V_u, name="g", graph=graph_)             # Circle Dirichlet boundary condition
noslip = fem.Function(V_u, name="noslip")        # No-slip homogenous Dirichlet boundary condition at the walls for the velocity
outflow = fem.Function(V_p, name="outflow")       # Outflow homogeneous Dirichlet boundary condition for the pressure
outflow.interpolate(lambda x: 0.0*x[0]+0.0)

dofs_walls = fem.locate_dofs_topological((V.sub(0), V_u), 1, ft.indices[ft.values == wall_marker])
dofs_inflow = fem.locate_dofs_topological((V.sub(0), V_u), 1, ft.indices[ft.values == inlet_marker])
dofs_outflow = fem.locate_dofs_topological((V.sub(1), V_p), 1, ft.indices[ft.values == outlet_marker])
dofs_obstacle = fem.locate_dofs_topological((V.sub(0), V_u), 1, ft.indices[ft.values == obstacle_marker])

bc_dofs_total = np.concatenate([dofs_walls[0], dofs_inflow[0], dofs_outflow[0], dofs_obstacle[0]])

bcs = [fem.dirichletbc(h, dofs_inflow, V.sub(0)),
        fem.dirichletbc(g, dofs_obstacle, V.sub(0), graph=graph_, map = V_u_map),
          fem.dirichletbc(noslip, dofs_walls, V.sub(0)),
            fem.dirichletbc(outflow, dofs_outflow, V.sub(1))]
# Parameters
nu = fem.Constant(mesh, ScalarType(1.0), name = "ν", graph = graph_)
alpha = 10.0
f = fem.Function(V_u, name="f")
f.interpolate(lambda x: (0.0 *x[0], 0.0 + 0.0*x[1]))

# Variational formulation
a = nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx\
    - ufl.div(v) * p * ufl.dx\
    + q * ufl.div(u) * ufl.dx
L = ufl.inner(f, v) * ufl.dx
F = a - L

# Define the problem solver
problem = fem.petsc.NonlinearProblem(F, up, bcs=bcs, graph=graph_)
solver = nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem, graph=graph_)

solver.solve(up, graph=graph_)   

# Define the objective function
dObs = ufl.Measure("ds", domain=mesh, subdomain_data=ft, subdomain_id=obstacle_marker)
J_form  = 0.5 * ufl.inner(ufl.grad(u), ufl.grad(u)) * ufl.dx + alpha / 2 * ufl.inner(g, g) * dObs

J = fem.assemble_scalar(fem.form(J_form, graph=graph_), graph=graph_)

if __name__ == "__main__":
    graph_.print()

    dJdnu = graph_.backprop(id(J), id(nu))
    dJdg = graph_.backprop(id(J), id(g))

    print("J(u, p) = ", J)
    print("dJdnu = ", dJdnu)
    print("||dJdg||_L2 = ", np.sqrt(np.dot(dJdg, dJdg)))


import unittest
class TestStokes(unittest.TestCase):
    def test_Stokes_dJdnu(self):
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
        DG0 = fem.FunctionSpace(mesh, ("DG", 0))
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
        
        # We now need to apply the boundary conditions to the rhs of the adjoint problem
        for bc_dof in bc_dofs_total:
            dJdu[int(bc_dof)] = 0

        adjoint_solution = np.linalg.solve(dFdu.transpose(), - dJdu.transpose())

        gradient = adjoint_solution.transpose() @ dFdnu + dJdnu

        self.assertTrue(np.allclose(graph_.backprop(id(J), id(nu)), gradient))


    def test_Stokes_dJdg(self):
        """
        In this test we calculate the derivative of the objective function
        with respect to the Dirichlet boundary condition g.
        
            dJ(u,p,g)/dg = ∂J/∂up ∂up/∂g + ∂J/∂g

        We can easily calculate the derivative of 
        ∂J/∂up and ∂J/∂g by using the symbolic differentiation of UFL.
        
        Since u is defined on a mixed element space we need to take care of
        the function spaces while calculating the derivative.

        In addition we know that 0 = F(u, p) and thus by deriving for g and using
        the combined ansatz space with up ∈ V we get
            0 = ∂F/∂up ∂up/∂g + ∂F/∂g
        However in this setting the Function spaces do not match anymore since g is
        defined on V_u and up on V = V_u x V_p.

        Thus we need to map the derivative ∂F/∂g from V_u to V.
        
        Putting it all together results in the adjoint approach
        of first calculating
            (∂F/∂up)ᵀ λ = - (∂J/∂up)ᵀ
        and then inserting it into
            dJdg = λᵀ ∂F/∂g + ∂J/∂g
        
        To get only the gradient values on the boundary we multiply the gradient
        with an identity matrix that is nonzero everywhere except on the boundary.

        The resulting function is still an element of the mixed element space,
        thus we need to map it back to the function space of g.
        """
        argument = ufl.TrialFunction(V)
        dJdu = ufl.derivative(J_form, up, argument) 

        dJdg = ufl.derivative(J_form, g)

        argument = ufl.TrialFunction(V)
        dFdu = ufl.derivative(F, up, argument)

        dJdu = fem.assemble_vector(fem.form(dJdu)).array
        dJdg = fem.assemble_vector(fem.form(dJdg)).array

        dFdg = fem.assemble_matrix(problem._a).to_dense()
        dFdu = fem.assemble_matrix(fem.form(dFdu), bcs = bcs).to_dense()

        # We now need to apply the boundary conditions to the rhs of the adjoint problem
        for bc_dof in bc_dofs_total:
            dJdu[int(bc_dof)] = 0

        adjoint_solution = np.linalg.solve(dFdu.transpose(), -dJdu.transpose())

        gradient =  adjoint_solution.transpose() @ dFdg 

        matrix = np.zeros((len(gradient), len(gradient)))
        for index in dofs_obstacle[0]:
            matrix[index, index] = 1.0

        gradient = (matrix @ gradient)[V_u_map] + dJdg
        self.assertTrue(np.allclose(gradient, graph_.backprop(id(J), id(g))))


