"""
In this demo we are interested in time-dependent problems namely the heat-equation

    du/dt - ∇u = 0
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
import numpy as np
import scipy.sparse as sps

from mpi4py import MPI
from dolfinx import mesh, fem, io, nls
from dolfinx_adjoint import *

from petsc4py.PETSc import ScalarType

import ufl

domain = mesh.create_unit_square(MPI.COMM_WORLD, 50, 50, mesh.CellType.triangle)
V = fem.FunctionSpace(domain, ("CG", 1))

dt = 0.001
T = 0.1

"""
Create true data set
"""
# Set the initial values of the temperature variable u
true_initial = fem.Function(V, name="u_true_initial")
true_initial.interpolate(lambda x: np.sin(2 * np.pi * x[0]) * np.sin(2 * np.pi * x[1]))
u_prev = true_initial.copy()
u_next = true_initial.copy()

v = ufl.TestFunction(V)
dt_constant = fem.Constant(domain, ScalarType(dt))

F = ufl.inner((u_next-u_prev)/dt_constant, v) * ufl.dx + ufl.inner(ufl.grad(u_next), ufl.grad(v)) * ufl.dx
problem = fem.petsc.NonlinearProblem(F, u_next)
solver = nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)

t = 0.0
xdmf = io.XDMFFile(domain.comm, "demo_Heat.xdmf", "w")
xdmf.write_mesh(domain)
xdmf.write_function(u_prev, t)

true_states = [u_prev.copy()]
times = [0.0]
while t < T:
    solver.solve(u_next)
    u_prev.vector[:] = u_next.vector[:]
    t += dt
    xdmf.write_function(u_prev, t)
    true_states.append(u_prev.copy())
    times.append(t)
xdmf.close()

"""
Create test data
"""

# Set the initial values of the temperature variable u
initial_guess = fem.Function(V, name="initial_guess")
initial_guess.interpolate(lambda x: 15.0 * x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1]))
u_prev = initial_guess.copy()
u_next = initial_guess.copy()

t = 0.0
test_states = [u.copy()]
xdmf = io.XDMFFile(domain.comm, "demo_Heat_test.xdmf", "w")
xdmf.write_mesh(domain)
xdmf.write_function(u)
while t < T:
    solver.solve(u_next)
    u.vector[:] = u_next.vector[:]
    t += dt
    test_states.append(u.copy())
    xdmf.write_function(u, t)
xdmf.close()

J_test = fem.assemble_scalar(fem.form(ufl.inner(true_initial-initial_guess, true_initial-initial_guess) * ufl.dx))

print("J(u) = ", J_test)
alpha = fem.Constant(domain, ScalarType(1.0e-7))
combined = zip(true_states, test_states)
J_form = sum(ufl.inner(true - test, true - test) * ufl.dx for (true, test) in combined) + alpha * ufl.inner(ufl.grad(initial_guess), ufl.grad(initial_guess)) * ufl.dx

J = fem.assemble_scalar(fem.form(J_form))

print("J(u) = ", J)