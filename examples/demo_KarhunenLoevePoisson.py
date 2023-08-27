"""
In this demo we consider the parametric Poisson's equation with homogeneous Dirichlet boundary conditions
and spatially varying diffusion coefficient:
    - ∇ ⋅ (ν(x) ∇u(x)) = f(x)           in Ω
                  u(x) = 0              on ∂Ω \ Γ
where Ω = [0,1]² is the unit square, Γ is the right boundary of the domain and f is the given source term with constant value f ≡ 1.
We use the whole diffussion field as the parameter in our parameter study.

We define the objective functional J(u) as the mean integral of the solution over the right boundary Γ of the domain.
    J(u) = 1 / |Γ| ∫_Γ u(x) dx

The goal of this demo is to use the adjoint method in a parameter study with respect to the diffusion coefficient a(x).
In these parameter studies we rely on an efficient estimation of the gradient of the objective functional with respect to the diffusion coefficient.

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
import dolfinx
from dolfinx import mesh, fem, io, nls
from dolfinx_adjoint import *
from petsc4py.PETSc import ScalarType
import ufl 

# We first need to create a graph object to store the computational graph. 
# This is done explicitly to maintain the guideline of FEniCSx.
# Every function that is created with a graph object will be added to the graph
# and its gradient will be computed automatically. 
graph_ = graph.Graph()

# Define mesh and finite element space
domain = mesh.create_unit_square(MPI.COMM_WORLD, 10, 10, mesh.CellType.triangle)
V = fem.FunctionSpace(domain, ("CG", 1))                
W = fem.FunctionSpace(domain, ("DG", 0))

# Define the basis functions and parameters
uh = fem.Function(V, name="uₕ", graph=graph_)                             
v = ufl.TestFunction(V)
f = fem.Constant(domain, ScalarType(1.0))  

# # Define the diffusion field as a truncated Karhunen Loeve expansion with the eigenpairs 
# # of the correlation matrix as the nodes of the expansion
# vertices = domain.geometry.x
# beta = 0.01
# num_points = np.shape(vertices)[0]

# # Define the correlation matrix
# correlation_matrix = np.zeros((num_points, num_points))
# for i in range(num_points):
#     for j in range(num_points):
#         correlation_matrix[i, j] = np.exp(-beta * np.linalg.norm(vertices[i] - vertices[j])**2)

# # Calculate the eigenpairs of the correlation matrix
# eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)

# Define diffusion field as a Karhunen Loeve expansion
nu = fem.Function(W, name="ν", graph=graph_)
nu.interpolate(lambda x: 1.0 + 0.0 * x[0])   

# Define the variational form and the residual equation
a = nu * ufl.inner(ufl.grad(uh), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx
F = a - L

# Define the boundary and the boundary conditions
domain.topology.create_connectivity(domain.topology.dim -1, domain.topology.dim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
uD = fem.Function(V, name="u_D")                          
uD.interpolate(lambda x: 1.0 + 0.0 * x[0])   
        
def bc_marker(x):
    return np.logical_or(np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0)), np.isclose(x[1], 1.0))
boundary_dofs = fem.locate_dofs_geometrical(V, lambda x: bc_marker(x))
bcs = [fem.dirichletbc(uD, boundary_dofs)]

# Define the problem solver and solve it
problem = fem.petsc.NonlinearProblem(F, uh, bcs=bcs, graph = graph_)             
solver = nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)

# In order to perform multiple runs through the forward model we put everything in a function
def forward(nu_vector = None, save_output = False):
    if nu_vector is not None:
        nu.vector.array[:] = nu_vector
    solver.solve(uh)                                                

    # Define the objective function as the integral over the right boundary
    # J(u) = 1 / |Γ| ∫_Γ u(x) dx
    c = fem.Function(V, name="c")
    c.interpolate(lambda x: np.isclose(x[0], 1.0))
    J = fem.assemble_scalar(fem.form(ufl.inner(c, uh) * ufl.dx, graph = graph_), graph = graph_)   
    dJdnu = graph_.backprop(id(J), id(nu))
    if save_output:
        with io.XDMFFile(MPI.COMM_WORLD, "output/forward_solution.xdmf", "w") as outfile:
            outfile.write_mesh(domain)
            outfile.write_function(uh)
            outfile.write_function(nu)
    return J, dJdnu   

print(forward()[0])

# We can now perform the parameter study using the active subspace method
# We first define the parameter space as a random uniform distribution in the interval [-1, 1]

m = np.shape(nu.vector.array)[0] # Dimension of the parameter space
n = 200                          # Number of samples
samples = np.random.uniform(-1, 1, (n, m))

# We now perform the parameter study
J_values = np.zeros(n)
dJdnu_values = np.zeros((n, m))

for i in range(n):
    J_values[i], dJdnu_values[i] = forward(samples[i])

covariance_matrix = np.dot(dJdnu_values.T, dJdnu_values)/n

e, eigenvectors = np.linalg.eigh(covariance_matrix)
e = abs(e)
idx = np.argsort(e)[::-1]
e = e[idx]
eigenvectors = eigenvectors[:,idx]
normalization = np.sign(eigenvectors[0,:])
normalization[normalization == 0] = 1
eigenvectors = eigenvectors * normalization

print(np.shape(eigenvectors), np.shape(e))
k = 10
import matplotlib.pyplot as plt
plt.figure()
plt.plot(e[:k], 'o')
plt.xlabel("Eigenvalue index")
plt.ylabel("Eigenvalue")
plt.title("Eigenvalues of the covariance matrix")
plt.savefig("output/eigenvalues.png")

with io.XDMFFile(MPI.COMM_WORLD, "output/eigenvectors.xdmf", "w") as outfile:
    eigenvector = fem.Function(W, name="eigenvector")
    outfile.write_mesh(domain)
    for i in range(k):
        eigenvector.vector[:] = eigenvectors[:,i]
        outfile.write_function(eigenvector, i)

# Use the eigenvectors to recalculate the cost function
print(forward(eigenvectors[:,0], save_output=True)[0])
print(forward(eigenvectors[:,1], save_output=True)[0])


