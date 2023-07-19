"""
In this demo we consider the parametric Poisson's equation with homogeneous Dirichlet boundary conditions:
    - ν ∆u = f              in Ω
         u = 0              on ∂Ω
where Ω = [0,1]² is the unit square, ν = 1 is the diffusion coefficient and f is a parameter.
We define the objective function (or quantity of interest)
    J(u) = ½∫_Ω ||u - g|| dx
where g is a known function.

The goal of this demo is to compute the gradient of J(u) with respect to f using the adjoint method.

We first start with the forward approach to solving the parametric Poisson's equation.
The weak formulation of the problem reads: find u ∈ V such that
    a(u, v) = L(v) ∀ v ∈ V
where
    a(u, v) = ∫_Ω ν ∇u · ∇v dx
    L(v)    = ∫_Ω f v dx.

We solve this problem with a residual equation
    F(u) = a(u, v) - L(v) != 0
"""

import numpy as np
import scipy.sparse as sps

from mpi4py import MPI
from dolfinx import mesh, fem, io, nls
from dolfinx_adjoint import *
from petsc4py.PETSc import ScalarType
import ufl 

# Define mesh and finite element space
domain = mesh.create_unit_square(MPI.COMM_WORLD, 64, 64, mesh.CellType.triangle)
V = fem.FunctionSpace(domain, ("CG", 1))                
W = fem.FunctionSpace(domain, ("DG", 0))

# Define the boundary and the boundary conditions
domain.topology.create_connectivity(domain.topology.dim -1, domain.topology.dim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
uD = fem.Function(V, name="u_D")                            # Overloaded
uD.interpolate(lambda x: 0.0 * x[0])                    # Should possibly be overloaded
boundary_dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1, boundary_facets)
bc = fem.dirichletbc(uD, boundary_dofs)                     # Should possibly be overloaded

# Define the basis functions and parameters
uh = fem.Function(V, name="uₕ")                             # Overloaded
v = ufl.TestFunction(V)
f = fem.Function(W, name="f")                               # Overloaded
nu = fem.Constant(domain, ScalarType(1.0))              # Should possibly be overloaded
f.interpolate(lambda x: x[0] + x[1])                    # Should possibly be overloaded

# Define the variational form and the residual equation
a = nu * ufl.inner(ufl.grad(uh), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx
F = a - L

# Define the problem solver and solve it
problem = fem.petsc.NonlinearProblem(F, uh, bcs=[bc])             # Overloaded
solver = nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)          # Overloaded
solver.solve(uh)                                        # Overloaded

# Define profile g
g = fem.Function(W, name="g")                               # Overloaded
g.interpolate(lambda x: 1 / (2 * np.pi**2) * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))   
# Define the objective function
J_form = 0.5 * ufl.inner(uh - g, uh - g) * ufl.dx
J = fem.assemble_scalar(fem.form(J_form))                       # Overloaded
print("J(u) = ", J)


print("\n-----------Adjoint approach-----------\n")
visualise()
# Test if it can compute the gradient of J with respect to uh
print(compute_gradient(J, f))


"""
We now turn to the adjoint approach to calculating the gradient of J(u) with respect to f.
Therefore, we first define a new reduced functional R(f) = J(u(f),f) and take the derivative with respect to f:
    dJ/df = dR/df = ∂J/∂u * du/df + ∂J/∂f                       (1)

The first term ∂J/∂u and the last term ∂J/∂f are easy to compute.
The second term du/df will be handled using the adjoint problem.

By taking the derivative of F(u) = 0 with respect to f, we obtain a representation of du/df:
    dF/df = ∂F/∂u * du/df + ∂F/∂f = 0
    => du/df = -(∂F/∂u)^-1 * ∂F/∂f                              (2)

Inserting (2) into (1) yields

    dJ/df = - ∂J/∂u * (∂F/∂u)^-1 * ∂F/∂f + ∂J/∂f                (3)

Our adjoint approach thus is to solve the adjoint problem
    ∂Fᵀ/∂u * λ =  - ∂Jᵀ/∂u                                      (4)
and then compute the gradient of J(u) with respect to f using (3).
    dJ/df = λᵀ * ∂F/∂f + ∂J/∂f                                  (5)   

In our case ∂J/∂f = 0, so we only need to compute λᵀ * ∂F/∂f.
"""
def gradient():
    dFdu = fem.assemble_matrix(fem.form(ufl.derivative(F, uh)), bcs=[bc])
    dFdu.finalize()
    shape = (V.dofmap.index_map.size_global, V.dofmap.index_map.size_global)
    dFduSparse = sps.csr_matrix((dFdu.data, dFdu.indices, dFdu.indptr), shape=shape).transpose()

    dJdu = - fem.assemble_vector(fem.form(ufl.derivative(J_form, uh))).array.transpose()

    adjoint_solution = sps.linalg.spsolve(dFduSparse, dJdu)

    dFdf = fem.assemble_matrix(fem.form(ufl.derivative(F, f))).to_dense()

    gradient = adjoint_solution.transpose() @ dFdf

    dJ = fem.Function(W, name="dJdf")
    dJ.vector.setArray(gradient)
    return dJ

dJ = gradient()

print(dJ.vector.array[:])

exit()

with io.XDMFFile(domain.comm, "demo_Poisson.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh, 0.0)
    xdmf.write_function(dJ, 0.0)

"""
In order to verify the compute gradient we perform a convergence study.
Using a Taylor expansion of R(f) around a perturbation δf, we obtain
    || R(f + ϵδf) - R(f)|| → 0 with O(ϵ).
However, with the gradient we obtain quadratic convergence
    || R(f + ϵδf) - R(f) - ϵ dR/df ⋅ δf|| → 0 with O(ϵ²).
"""

# Define the perturbation
delta = fem.Function(W, name="δf")
delta.interpolate(lambda x: x[0] * x[1])
J_value = fem.assemble_scalar(fem.form(J_form))

values_linear = []
values_quadratic = []
for epsilon in np.linspace(1, 0, 10):
    f.interpolate(lambda x: x[0] + x[1] + epsilon * (x[0] * x[1]))
    solver.solve(uh)
    J_eps = fem.assemble_scalar(fem.form(J_form))
    dJ = gradient()

    values_linear.append(np.abs(J_eps - J_value))
    values_quadratic.append(np.abs(J_eps - J_value - epsilon * dJ.vector.array.transpose() @ delta.vector.array))

print("Linear convergence: ", values_linear)
print("Quadratic convergence: ", values_quadratic)

import matplotlib.pyplot as plt
plt.figure(tight_layout=True)
plt.plot(values_linear, label="linear")
plt.plot(values_quadratic, label="quadratic")
plt.xticks([0, 9], [1, 0])
plt.legend()
plt.show()

