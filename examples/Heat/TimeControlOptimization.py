"""
In this demo we are considering the heat-equation with a time-dependent source term f(x,t) and homogeneous Dirichlet boundary values

    du/dt - ν Δu = f      in Ω × (0,T]
             u = 0      for Ω × {0}
             u = 0      for ∂Ω × (0,T]

where Ω is the unit square, T is the final time, u is the unknown temperature variation and ν is the termal diffusivity.

For the initial temperature we choose

    g(x,y) = sin(2πx) sin(2πy)

In order to solve this problem we employ a Rothe method (method of lines), where the weak formulation
is given by

    ∫_Ω du/dt v dx + ν ∫_Ω ∇u ⋅ ∇v dx = ∫_Ω f v dx

for all v∈ H\^1(Ω). The time derivative is approximated by a backward Euler scheme

    ∫_Ω (u-uₙ)/δₜ v dx + ν ∫_Ω ∇u ⋅ ∇v dx = ∫_Ω f v dx

where uₙ is the solution at the previous time step. 

The objective function in this case is the integral over the time of the L²-norm of the difference between the 
temperature and a known temperature profile.

    J(u) = ∫_0^T ∫_Ω (u - u_d)² dx dt

where u_d is the known temperature profile.

We approximate the time-integral with an trapezoidal rule

    J(u) = ∫₀ᵀ ∫_Ω (u - u_d)² dx dt ≈ ∑ᵢ₌₀ᴺ⁻¹ 0.5 * δₜ(∫_Ω (uᵢ - u_d)² dx +  ∫_Ω (uᵢ₊₁ - u_d)² dx)
         = 0.5 * δₜ ∫_Ω (u₀ - u_d)² dx + ∑ᵢ₌₁ᴺ⁻¹ δₜ ∫_Ω (uᵢ - u_d)² dx + 0.5 * δₜ ∫_Ω (u_N - u_d)² dx
"""
import numpy as np
import scipy.sparse as sps

from mpi4py import MPI
from dolfinx import mesh, fem, io, nls
from dolfinx_adjoint import *

from petsc4py.PETSc import ScalarType

import ufl
# Create graph object
graph_ = graph.Graph()

domain = mesh.create_unit_square(MPI.COMM_WORLD, 50, 50, mesh.CellType.triangle)
V = fem.FunctionSpace(domain, ("CG", 1))

domain.topology.create_connectivity(domain.topology.dim -1, domain.topology.dim)
boundary_facets = mesh.locate_entities_boundary(
    domain, domain.topology.dim -1, lambda x: np.full(x.shape[1], True, dtype=bool))
bc = fem.dirichletbc(ScalarType(0), fem.locate_dofs_topological(V, domain.topology.dim -1, boundary_facets), V)

T = 0.5
dt = 0.1
dt_constant = fem.Constant(domain, ScalarType(dt))    

nu = fem.Constant(domain, ScalarType(1.0e-5))

v = ufl.TestFunction(V)
u_prev = fem.Function(V, name = "u_prev", graph=graph_)
u = fem.Function(V, name = "u_next", graph=graph_)

f = fem.Function(V, name = "source", graph = graph_)
f.interpolate(lambda x: 1.0 + 0.0*x[0])

F = ufl.inner((u-u_prev)/dt_constant, v) * ufl.dx + nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - f * v * ufl.dx

u_iterations = [u.copy(name = f"u_0")]
times = [0.0]
t = dt
i = 1
f_sources=[]
while t <= T:
    f_source = fem.Function(V, name = "control_source_"+str(i), graph = graph_)
    f_source.interpolate(lambda x: 1.0 + 0.0*x[0])
    f_sources.append(f_source)
    f.assign(f_source, graph = graph_, version = i)
    problem = fem.petsc.NonlinearProblem(F, u, bcs = [bc], graph=graph_)
    solver = nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem, graph = graph_)
    solver.solve(u, graph = graph_, version = i)
 
    u_prev.assign(u, graph = graph_, version = i)

    u_iterations.append(u.copy(name = f"u_{i}", graph=graph_))
    times.append(t)

    t += dt
    i += 1


data = fem.Function(V, name = "data")

J_form = 0.0
data_iterations = []
for i, u_i in enumerate(u_iterations):
    # Apply trapezoidal rule
    if i == 0 or i == len(u_iterations) - 1:
        weight = 0.5
    else:
        weight = 1.0

    data.interpolate(lambda x: 16*x[0]*(x[0]-1)*x[1]*(x[1]-1)*np.sin(np.pi*times[i]))
    data_i = data.copy(name = f"data_{i}")
    data_iterations.append(data_i)
    J_form += weight * dt_constant * ufl.inner(u_i - data_i , u_i - data_i) * ufl.dx


J = fem.assemble_scalar(fem.form(J_form, graph = graph_), graph = graph_)



if __name__ == "__main__":
    # Visualize the graph
    graph_.visualise(style="shell")

    # Compute the gradient of the objective function J with respect to the control source terms
    dJdf = []
    for i, f_control in enumerate(f_sources):
        dJdf.append(graph_.backprop(id(J), id(f_control)))

    print("J = ", J)

    # Visualization of the results
    dJdf_function = fem.Function(V, name = "dJdf")
    u_vis_function = fem.Function(V, name = "temperature")
    data_function = fem.Function(V, name = "data")

    u_vis_function.vector[:] = u_iterations[0].vector[:]
    data_function.vector[:] = data_iterations[0].vector[:]

    xdmf = io.XDMFFile(MPI.COMM_WORLD, "time_control.xdmf", "w")
    xdmf.write_mesh(domain)
    xdmf.write_function(dJdf_function)
    xdmf.write_function(u_vis_function)
    xdmf.write_function(data_function)

    for i, dJdf_i in enumerate(dJdf):
        print(f"||dJdf_{i}||_L2 ={np.sqrt(np.dot(dJdf_i, dJdf_i))}")
        
        dJdf_function.vector[:] = dJdf_i
        u_vis_function.vector[:] = u_iterations[i+1].vector[:]
        data_function.vector[:] = data_iterations[i+1].vector[:]

        xdmf.write_function(data_function, times[i+1])
        xdmf.write_function(dJdf_function, times[i+1]) # Index of the source terms start at 0 for time δₜ
        xdmf.write_function(u_vis_function, times[i+1])
    xdmf.close()