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

graph_ = graph.Graph()

# We first need to create a graph object to store the computational graph. 
# This is done explicitly to maintain the guideline of FEniCSx.
# Every function that is created with a graph object will be added to the graph
# and its gradient will be computed automatically. 

# Mesh parameters
gmsh.initialize()
L = 30
H = 10
c_x = 10
c_y = 5
r = 2.5
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
    gmsh.model.mesh.field.setNumber(2, "LcMin", 0.2)
    gmsh.model.mesh.field.setNumber(2, "LcMax", 0.8)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0.3*r)
    gmsh.model.mesh.field.setNumber(2, "DistMax", r)
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

dofs_walls = fem.locate_dofs_topological((V.sub(0), V_u), 1, ft.indices[ft.values == wall_marker])
dofs_inflow = fem.locate_dofs_topological((V.sub(0), V_u), 1, ft.indices[ft.values == inlet_marker])
dofs_outflow = fem.locate_dofs_topological((V.sub(1), V_p), 1, ft.indices[ft.values == outlet_marker])
dofs_obstacle = fem.locate_dofs_topological((V.sub(0), V_u), 1, ft.indices[ft.values == obstacle_marker])

g_boundary_indices = []
for dof in dofs_obstacle[0]:
    g_boundary_indices.append(V_u_map.index(dof))

vq = ufl.TestFunction(V)
(v, q) = ufl.split(vq)
# Boundary conditions   
h = fem.Function(V_u, name="f")                                     # Inflow Dirichlet boundary condition
h.interpolate(lambda x: np.stack((x[1]*(10-x[1])/25, 0.0* x[0])))
noslip = fem.Function(V_u, name="noslip")                           # No-slip homogenous Dirichlet boundary condition at the walls for the velocity
outflow = fem.Function(V_p, name="outflow")                         # Outflow homogeneous Dirichlet boundary condition for the pressure
outflow.interpolate(lambda x: 0.0*x[0]+0.0)

# Parameters
nu = fem.Constant(mesh, ScalarType(1.0), name = "ν")
alpha = 10.0
f = fem.Function(V_u, name="f")
f.interpolate(lambda x: (0.0 *x[0], 0.0 + 0.0*x[1]))

dObs = ufl.Measure("ds", domain=mesh, subdomain_data=ft, subdomain_id=obstacle_marker)

dim = len(dofs_obstacle[0])

g_vis_function = fem.Function(V_u, name="g_vis")
xdmf = io.XDMFFile(MPI.COMM_WORLD, "stokes_optimization.xdmf", "w")
convergence_data = []
xdmf.write_mesh(mesh)
vis_index = 0
def forward(g_array, plot = False):
    global vis_index
    graph_.clear()
    
    up = fem.Function(V, name="up", graph=graph_)
    (u, p) = ufl.split(up)

    # Circle Dirichlet boundary condition
    g = fem.Function(V_u, name="g", graph=graph_)  
    for i, index in enumerate(g_boundary_indices):
        g.vector.array[index] = g_array[i]
    bcs = [fem.dirichletbc(h, dofs_inflow, V.sub(0)),
            fem.dirichletbc(g, dofs_obstacle, V.sub(0), graph=graph_, map = V_u_map),
            fem.dirichletbc(noslip, dofs_walls, V.sub(0)),
                fem.dirichletbc(outflow, dofs_outflow, V.sub(1))]

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
    J_form  = 0.5 * ufl.inner(ufl.grad(u), ufl.grad(u)) * ufl.dx + 0.5 * alpha * ufl.inner(g, g) * dObs

    J = fem.assemble_scalar(fem.form(J_form, graph=graph_), graph=graph_)
    if plot:
        u, p = up.split()
        u.name = "velocity"
        p.name = "pressure"
        xdmf.write_function(u, vis_index)
        xdmf.write_function(p, vis_index)
        xdmf.write_function(g, vis_index)
        convergence_data.append(J)
        vis_index += 1

    dJdg = graph_.backprop(id(J), id(g))
    gradient = dJdg.array[g_boundary_indices]
    
    print(J)
    return J, gradient
xdmf.close()

forward_callback = lambda g_array: forward(g_array, plot = True)

g_array = np.zeros(dim)

from scipy.optimize import minimize

forward(g_array, plot = True)

res = minimize(forward, g_array, method = "BFGS", jac=True, tol=1e-8,callback=forward_callback,
                 options={"gtol": 1e-8, 'disp':True, 'maxiter':100})

import matplotlib.pyplot as plt
plt.figure()
plt.plot(convergence_data)
plt.xlabel("Iteration")
plt.ylabel("Objective function")
plt.grid(True)
plt.savefig("stokes_optimization_convergence.pdf")

plt.show()
