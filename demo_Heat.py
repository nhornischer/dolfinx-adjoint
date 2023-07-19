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

domain = mesh.create_unit_square(MPI.COMM_WORLD, 64, 64, mesh.CellType.triangle)
V = fem.FunctionSpace(domain, ("CG", 1))

dt = 0.001
T = 0.1


"""
Create data set
"""
# Set the initial values of the temperature variable u
u = fem.Function(V, name="u")
u.interpolate(lambda x: np.sin(2 * np.pi * x[0]) * np.sin(2 * np.pi * x[1]))
u_old = u.copy()

v = ufl.TestFunction(V)
dt_constant = fem.Constant(domain, ScalarType(dt))

F = ufl.inner((u-u_old)/dt_constant, v) * ufl.dx + ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
problem = fem.petsc.NonlinearProblem(F, u)
solver = nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)

xdmf = io.XDMFFile(domain.comm, "demo_Heat.xdmf", "w")
xdmf.write_mesh(domain)
xdmf.write_function(u)

t = 0.0
while t < T:
    solver.solve(u)
    u_old.vector[:] = u.vector[:]
    t += dt
    xdmf.write_function(u, t)
xdmf.close()

J = fem.assemble_scalar(fem.form(ufl.inner(u, u) * ufl.dx))

print("J(u) = ", J)

visualise()