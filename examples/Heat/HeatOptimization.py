"""
In this demo we are interested in time-dependent problems namely the heat-equation

    du/dt - Δu = 0
             u = g for Ω × {0}

where Ω is the unit square, u is the temperature and g is the initial temperature.
The problem has homogeneous Neumann boundary conditions.
For the initial temperature we choose

    g(x,y) = sin(2πx) sin(2πy)

In order to solve this problem we employ a Rothe method (method of lines), where the weak formulation
is given by

    ∫_Ω du/dt v dx + ∫_Ω ∇u ⋅ ∇v dx = 0

for all v∈ H\^1(Ω). The time derivative is approximated by a backward Euler scheme

    ∫_Ω (u-u\^n)/dt v dx + ∫_Ω ∇u ⋅ ∇v dx = 0

where u\^n is the solution at the previous time step. 

The objective function in our case is the L\^2-norm of the temperature at the last time step.

    J(u) = ∫_Ω u ⋅ u dx
"""

from memory_profiler import profile
import numpy as np
import scipy.sparse as sps

from mpi4py import MPI
from dolfinx import mesh, fem, io, nls
from dolfinx_adjoint import *

from petsc4py.PETSc import ScalarType

import ufl
comm = MPI.COMM_WORLD

domain = mesh.create_unit_square(MPI.COMM_WORLD, 32, 32, mesh.CellType.triangle)
V = fem.FunctionSpace(domain, ("CG", 1))

dt = 0.01
T = 0.04
"""
Create true data set
"""
# Set the initial values of the temperature variable u
true_initial = fem.Function(V, name="true initial value")
true_initial.interpolate(lambda x: np.sin(2 * np.pi * x[0]) * np.sin(2 * np.pi * x[1]))
u_prev = true_initial.copy(name = "u_prev")
u_next = true_initial.copy(name = "g(t)")

v = ufl.TestFunction(V)
dt_constant = fem.Constant(domain, ScalarType(dt))    

F = ufl.inner((u_next-u_prev)/dt_constant, v) * ufl.dx + ufl.inner(ufl.grad(u_next), ufl.grad(v)) * ufl.dx
problem = fem.petsc.NonlinearProblem(F, u_next)
solver = nls.petsc.NewtonSolver(comm, problem)

t = 0.0
true_states = [u_prev.copy()]
times = [0.0]

xdmf = io.XDMFFile(comm, "heat_opt_g.xdmf", "w")
xdmf.write_mesh(domain)
xdmf.write_function(u_next, t)
while t < T:
    solver.solve(u_next)
    u_prev.vector[:] = u_next.vector[:]
    t += dt
    xdmf.write_function(u_next, t)
    if t >= T/2:
        true_states.append(u_next.copy())
        times.append(t)
xdmf.close()

alpha = fem.Constant(domain, ScalarType(1.0e-7))
"""
Create test data
"""
# Create graph object
graph_ = graph.Graph()
convergence_data = []
inital_vis = fem.Function(V, name = "initial value")
def fun(initial_array, callback = False, plot = False):
    if plot:
        filename = "heat_opt_u_" + str(len(convergence_data)) + ".xdmf"
        xdmf = io.XDMFFile(comm, filename, "w")
        xdmf.write_mesh(domain)
    graph_.clear()
    # Set the initial values of the temperature variable u
    initial_guess = fem.Function(V, name = "initial_guess", graph=graph_)
    initial_guess.vector.array[:] = initial_array
    u_prev = initial_guess.copy(graph=graph_, name = "u_prev")
    u_next = initial_guess.copy(graph=graph_, name = "u(t)")

    F = ufl.inner((u_next-u_prev)/dt_constant, v) * ufl.dx + ufl.inner(ufl.grad(u_next), ufl.grad(v)) * ufl.dx
    t = 0.0
    test_states = [u_next.copy(graph=graph_, name = "temperature_" + str(0))]
    i = 0
    if plot:
        xdmf.write_function(u_next, t)
    while t < T:
        i += 1
        F = ufl.inner((u_next-u_prev)/dt_constant, v) * ufl.dx + ufl.inner(ufl.grad(u_next), ufl.grad(v)) * ufl.dx
        problem = fem.petsc.NonlinearProblem(F, u_next, graph=graph_)
        solver = nls.petsc.NewtonSolver(comm, problem, graph = graph_)

        solver.solve(u_next, graph=graph_, version = i)
        t += dt
        u_prev.assign(u_next, graph = graph_, version = i)
        if t>= T/2:
            test_states.append(u_next.copy(graph=graph_, name = "temperature_" + str(i)))
        if plot:
            xdmf.write_function(u_next, t)
        
    combined = zip(true_states, test_states)
    J_form = sum(ufl.inner(true - test, true - test) * ufl.dx for (true, test) in combined) + alpha * ufl.inner(ufl.grad(initial_guess), ufl.grad(initial_guess)) * ufl.dx
    J = fem.assemble_scalar(fem.form(J_form, graph=graph_), graph=graph_)

    dJdinit = graph_.backprop(id(J), id(initial_guess)).array[:]
    print(J)
    if plot:
        graph_.visualise()
        xdmf.close()
    if callback:
        convergence_data.append(J)
    del initial_guess, u_prev, u_next, F, problem, solver, combined, J_form, test_states

    return J, dJdinit


initial_guess = fem.Function(V, name = "initial_guess", graph=graph_)
initial_guess.interpolate(lambda x: 15.0 * x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1]))

initial_values = initial_guess.vector.array[:]

forward_callback = lambda g_array: fun(g_array, callback = True)

from scipy.optimize import minimize

fun(initial_values, plot = True)
res = minimize(fun, initial_values, method = "BFGS", jac=True, tol=1e-4, callback=forward_callback,
                 options={"gtol": 1e-4, 'disp':True, 'maxiter':100})

fun(res.x, plot = True)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(convergence_data)
plt.xlabel("Iteration")
plt.ylabel("Objective function")
plt.grid(True)
plt.savefig("heat_optimization_convergence.pdf")