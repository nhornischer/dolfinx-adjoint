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
import numpy as np
import scipy.sparse as sps

from mpi4py import MPI
from dolfinx import mesh, fem, io, nls
from dolfinx_adjoint import *

from petsc4py.PETSc import ScalarType

import ufl

domain = mesh.create_unit_square(MPI.COMM_WORLD, 32, 32, mesh.CellType.triangle)
V = fem.FunctionSpace(domain, ("CG", 1))

dt = 0.01
T = 0.3
"""
Create true data set
"""
# Set the initial values of the temperature variable u
true_initial = fem.Function(V, name="u_true_initial")
true_initial.interpolate(lambda x: np.sin(2 * np.pi * x[0]) * np.sin(2 * np.pi * x[1]))
u_prev = true_initial.copy(name = "u_prev")
u_next = true_initial.copy(name = "u_next")

v = ufl.TestFunction(V)
dt_constant = fem.Constant(domain, ScalarType(dt))    

F = ufl.inner((u_next-u_prev)/dt_constant, v) * ufl.dx + ufl.inner(ufl.grad(u_next), ufl.grad(v)) * ufl.dx
problem = fem.petsc.NonlinearProblem(F, u_next)
solver = nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)

t = 0.0
true_states = [u_prev.copy()]
times = [0.0]
while t < T:
    solver.solve(u_next)
    u_prev.vector[:] = u_next.vector[:]
    t += dt
    true_states.append(u_next.copy())
    times.append(t)

"""
Create test data
"""
# Create graph object
graph_ = graph.Graph()

# Set the initial values of the temperature variable u
initial_guess = fem.Function(V, name = "initial_guess", graph=graph_)
initial_guess.interpolate(lambda x: 15.0 * x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1]))
u_prev = initial_guess.copy(graph=graph_, name = "u_prev")
u_next = initial_guess.copy(graph=graph_, name = "u_next")

F = ufl.inner((u_next-u_prev)/dt_constant, v) * ufl.dx + ufl.inner(ufl.grad(u_next), ufl.grad(v)) * ufl.dx
t = 0.0
test_states = [u_next.copy(graph=graph_, name = "temperature_" + str(0))]
i = 0
while t < T:
    i += 1
    F = ufl.inner((u_next-u_prev)/dt_constant, v) * ufl.dx + ufl.inner(ufl.grad(u_next), ufl.grad(v)) * ufl.dx
    problem = fem.petsc.NonlinearProblem(F, u_next, graph=graph_)
    solver = nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem, graph = graph_)

    solver.solve(u_next, graph=graph_, version = i)
    t += dt
    u_prev.assign(u_next, graph = graph_, version = i)
    test_states.append(u_next.copy(graph=graph_, name = "temperature_" + str(i)))

alpha = fem.Constant(domain, ScalarType(1.0e-7))
combined = zip(true_states, test_states)
J_form = sum(ufl.inner(true - test, true - test) * ufl.dx for (true, test) in combined) + alpha * ufl.inner(ufl.grad(initial_guess), ufl.grad(initial_guess)) * ufl.dx
J = fem.assemble_scalar(fem.form(J_form, graph=graph_), graph=graph_)

if __name__ == "__main__":
    # Visualize the graph
    graph_.visualise()

    dJdinit = graph_.backprop(id(J), id(initial_guess))

    # Visualization of the results
    grad_func = fem.Function(V, name="gradient")
    grad_func.vector[:] = dJdinit

    temperature_function = fem.Function(V, name = "temperature")
    profile_function = fem.Function(V, name = "profile")
    xdmf = io.XDMFFile(MPI.COMM_WORLD, "heat_results.xdmf", "w")
    xdmf.write_mesh(domain)
    xdmf.write_function(grad_func, 0.0)
    for i, u in enumerate(test_states):
        temperature_function.vector[:] = u.vector[:]
        profile_function.vector[:] = true_states[i].vector[:]
        xdmf.write_function(temperature_function, t = times[i])
        xdmf.write_function(profile_function, t = times[i])
    xdmf.close()

    print("J(u) = ", J)    
    print("||dJdu_0||_L2 = ", np.sqrt(np.dot(dJdinit, dJdinit)))

import unittest
class TestHeat(unittest.TestCase):
    def test_Heat_inital(self):
        gradient = 0.0
        for i, u in enumerate(test_states):
            dJdu = ufl.derivative(J_form, u)
            dJdu = fem.assemble_vector(fem.form(dJdu)).array
            print(u.vector.array[:])
            if i == 0:
                gradient += dJdu
            else:
                start = dJdu
                for j in range(i, 0, -1):    
                    F_manipulated = ufl.replace(F, {u_next: u, u_prev: test_states[j-1]})
                    dFdu_next = ufl.derivative(F_manipulated, u)
                    dFdu_next = fem.assemble_matrix(fem.form(dFdu_next)).to_dense()
                    dFdu_prev = ufl.derivative(F_manipulated, test_states[j-1])
                    dFdu_prev = fem.assemble_matrix(fem.form(dFdu_prev)).to_dense()
                    adjoint_sol = np.linalg.solve(dFdu_next.transpose(), - start.transpose())
                    start = adjoint_sol @ dFdu_prev
                gradient += start
    
        self.assertTrue(np.allclose(gradient, graph_.backprop(id(J), id(initial_guess))))

    