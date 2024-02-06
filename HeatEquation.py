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
T = 0.02

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

# Set dirichlet boundary conditions
uD = fem.Function(V)
uD.interpolate(lambda x: 0.0 + 0.0 *x[0])
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

F = ufl.inner((u_next-u_prev)/dt_constant, v) * ufl.dx + ufl.inner(ufl.grad(u_next), ufl.grad(v)) * ufl.dx
problem = fem.petsc.NonlinearProblem(F, u_next)
solver = nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)

t = 0.0
while t < T:
    solver.solve(u_next)
    u_prev.vector[:] = u_next.vector[:]
    t += dt
true_data = u_next.copy()

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
i = 0

# We store the states for the whole time-domain in order to test the results later on and to visualise the results.
# These copies are not parts of the graph.
u_iterations = [u_next.copy()]
while t < T:
    i += 1
    F = ufl.inner((u_next-u_prev)/dt_constant, v) * ufl.dx + ufl.inner(ufl.grad(u_next), ufl.grad(v)) * ufl.dx
    problem = fem.petsc.NonlinearProblem(F, u_next, graph=graph_)
    solver = nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem, graph = graph_)

    solver.solve(u_next, graph=graph_, version = i)
    t += dt
    u_prev.assign(u_next, graph = graph_, version = i)

    # Storing the iterations for later visualisation and testing
    u_iterations.append(u_next.copy())

alpha = fem.Constant(domain, ScalarType(1.0e-6))

J_form = ufl.inner(true_data - u_next, true_data - u_next) * ufl.dx + alpha * ufl.inner(ufl.grad(initial_guess), ufl.grad(initial_guess)) * ufl.dx
J = fem.assemble_scalar(fem.form(J_form, graph=graph_), graph=graph_)

if __name__ == "__main__":
    # Visualize the graph
    graph_.print()

    dJdinit = graph_.backprop(id(J), id(initial_guess))

    # Visualization of the results
    grad_func = fem.Function(V, name="gradient")
    grad_func.vector[:] = dJdinit

    temperature_function = fem.Function(V, name = "temperature")

    print("J(u) = ", J)    
    print("||dJdu_0||_L2 = ", np.sqrt(np.dot(dJdinit, dJdinit)))

import unittest
class TestHeat(unittest.TestCase):
    def test_Heat_initial(self):
        """ 
        In this test we compare the automatically constructed adjoint equations and its solution with 
        the analytically derived adjoint equations and their discretized solution.

        We compute the derivative of J with respect to the initial guess u_0.
            dJ/du_0 = ∂J/∂u_N * du_N/du_{N-1} * ... * du_1/du_0 + ∂J/∂u_0                           (1.1)
        where u_i is the solution at time step i.

        The first and last term ∂J/∂u_N and ∂J/∂u_0 can be easily obtained by the unified form language (UFL).

        The second term du_N/du_{N-1} can be obtained by solving the adjoint equation. To this end, we take the derivative
        of the residual equation F(u_N, u_{N-1}) = 0 with respect to u_{N-1} and obtain 
            dF/du_{N-1} = ∂F/∂u_N * du_N/du_{N-1} + ∂F/∂u_{N-1} = 0

            => du_N/du_{N-1} = - (∂F/∂u_N)^{-1} * ∂F/∂u_{N-1}                                       (1.2)

        Inserting (1.2) into (1.1) yields
            dJ/du_0 = ∂J/∂u_N * -(∂F/∂u_N)^{-1} * ∂F/∂u_{N-1}) * ... * du_1/du_0 + ∂J/∂u_0

        This leads to the first adjoint equation
            (∂Fᵀ/∂u_N) * λ_N = - ∂Jᵀ/∂u_N
        and the subsequent adjoint equations
            (∂Fᵀ/∂u_{i-1}) * λ_{i-1} = - λ_i * ∂Fᵀ/∂u_{i-1}
        Here it is important that F depends on u_{i-1} and u_i, denoted with F_i
        
        We then obtain
            dJ/du_0 = λᵀ_1 * ∂F/∂u_{0} + ∂J/∂u_0
        """
        dJdu = ufl.derivative(J_form, u_next)
        dJdu_0 = ufl.derivative(J_form, initial_guess)

        dJdu_0 = fem.assemble_vector(fem.form(dJdu_0)).array
        dJdu = fem.assemble_vector(fem.form(dJdu)).array

        rhs = dJdu
        
        for i in range(len(u_iterations)-1, 0, -1):
            F_i = ufl.replace(F, {u_next: u_iterations[i], u_prev: u_iterations[i-1]})
            dF_idu_i = ufl.derivative(F_i, u_iterations[i])
            dF_idu_i = fem.assemble_matrix(fem.form(dF_idu_i)).to_dense()

            lambda_i = np.linalg.solve(dF_idu_i.transpose(), - rhs.transpose())

            dF_idu_i_1 = ufl.derivative(F_i, u_iterations[i-1])
            dF_idu_i_1 = fem.assemble_matrix(fem.form(dF_idu_i_1)).to_dense()
            rhs = lambda_i.transpose() @ dF_idu_i_1

        gradient = rhs + dJdu_0

        # Compare the results
        self.assertTrue(np.allclose(graph_.backprop(id(J), id(initial_guess)), gradient))
    