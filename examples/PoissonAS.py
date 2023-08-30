"""
In this demo we consider the corresponding optimization problem to the Poisson equation.

The goal of this demo is to find the source term f such that the 
solution of the Poisson equation is as close as possible to a 
given profile g.

We define the objective function (or quantity of interest)
    J(u) = ½∫_Ω ||u - g|| dx
where g is a known function.

The optimization problem is then to find f such that
    f = argmin J(u(f))
where u(f) is the solution of the Poisson equation with source term f.

The forward problem is defined as follows:
    - ν ∆u = f              in Ω
        u = 0              on ∂Ω
where Ω = [0,1]² is the unit square, ν = 1 is the diffusion coefficient and f is a parameter.

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

# Create mesh and define function space
domain = mesh.create_unit_square(MPI.COMM_WORLD, 64, 64, mesh.CellType.triangle)
V = fem.FunctionSpace(domain, ("CG", 1))  

# Define profile g
g = fem.Function(V, name="g")                                   
g.interpolate(lambda x: 1 / (2 * np.pi**2) * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))   

# Define the boundary and the boundary conditions
domain.topology.create_connectivity(domain.topology.dim -1, domain.topology.dim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
uD = fem.Function(V, name="u_D")                          
uD.interpolate(lambda x: 0.0 * x[0])   
        
boundary_dofs_L = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0))
boundary_dofs_R = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 1.0))
boundary_dofs_T = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[1], 1.0))
boundary_dofs_B = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[1], 0.0))

bcs = [fem.dirichletbc(uD, boundary_dofs_L),
    fem.dirichletbc(uD, boundary_dofs_R),
    fem.dirichletbc(uD, boundary_dofs_T),
    fem.dirichletbc(uD, boundary_dofs_B)]

v = ufl.TestFunction(V)
nu = fem.Constant(domain, ScalarType(1.0), name = "ν") 

alpha = fem.Constant(domain, ScalarType(1e-6), name = "α")      

            
def fun(f_array):
    graph_.clear()

    # Define the basis functions and parameters
    uh = fem.Function(V, name="uₕ", graph=graph_)                             
    
    f = fem.Function(V, name="f", graph=graph_)                               

    # Define the variational form and the residual equation
    a = nu * ufl.inner(ufl.grad(uh), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx
    F = a - L

    f.vector.array[:] = f_array
    # Define the problem solver and solve it
    problem = fem.petsc.NonlinearProblem(F, uh, bcs=bcs, graph = graph_)             
    solver = nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem, graph = graph_)          
    solver.solve(uh, graph = graph_)    
    
    # Define the objective function
    J_form = 0.5 * ufl.inner(uh - g, uh - g) * ufl.dx + 0.5 * alpha * f * f * ufl.dx
                                                  
    J = fem.assemble_scalar(fem.form(J_form, graph = graph_), graph = graph_)                       
    dJdf = graph_.backprop(id(J), id(f))
    return J, dJdf.array[:]

f_initial = fem.Function(V, name="f_initial")

m = len(f_initial.vector.array)     # Number of parameters
alpha = 2                           # Oversampling factor (originally 2)
k = m / 3                           # Number of eigenvalues of interest

M = int(alpha * k * np.log(m)) 
M = 300
k = 400

# Sample the parameter space
f_array = np.random.uniform(-1, 1, (M, m))

gradients = np.zeros([M, m])
J_values = np.zeros(M)

for i in range(M):
    print(f"{i}/{M}", end='\r')
    J_values[i], gradients[i, :] = fun(f_array[i])
print("")

# Compute the active subspace

weights = np.ones((M, 1))/M

covariance = np.dot(gradients.T, gradients * weights)

e, W = np.linalg.eigh(covariance)
e = abs(e)
idx = np.argsort(e)[::-1]
e = e[idx]
W = W[:,idx]
normalization = np.sign(W[0,:])
normalization[normalization == 0] = 1
W = W * normalization

import matplotlib.pyplot as plt
plt.figure()
plt.plot(e[:k], 'o')
plt.savefig("Poisson_eigenvalues.png")
plt.show()

xdmf = io.XDMFFile(MPI.COMM_WORLD, "PoissonAS.xdmf", "w")
xdmf.write_mesh(domain)
f_vis = fem.Function(V, name="f_vis")
# Show possible important directions
for i in range(310):
    f_vis.vector.array[:] = W[:,i]
    xdmf.write_function(f_vis, i)
xdmf.close()