"""
As a test example, we will model a clamped beam deformed under its own weigth in 3D.
This can be modeled, by setting the right-hand side body force per unit volume to 
$f=(0,0,-\rho g)$ with $\rho$ the density of the beam and $g$ the acceleration of gravity.
The beam is box-shaped with length $L$ and has a square cross section of width $W$.
We set $u=u_D=(0,0,0)$ at the clamped end, x=0. The rest of the boundary is traction free,
 that is, we set $T=0$. We start by defining the physical variables used in the program.

"""
# Scaled variable
L = 1
W = 0.2
mu = 1
rho = 1
delta = W/L
gamma = 0.4*delta**2
beta = 1.25
lambda_ = beta
g = gamma

"""
We then create the mesh, which will consist of hexahedral elements, along with the function space.
 We will use the convenience function `VectorFunctionSpace`. 
 However, we also could have used `ufl`s functionality, creating a vector element `element = ufl.VectorElement("CG", mesh.ufl_cell(), 1)
`, and intitializing the function space as `V = dolfinx.fem.FunctionSpace(mesh, element)`.

"""
import numpy as np
import ufl

from mpi4py import MPI
from petsc4py.PETSc import ScalarType

from dolfinx import mesh, fem, plot, io

domain = mesh.create_box(MPI.COMM_WORLD, [np.array([0,0,0]), np.array([L, W, W])],
                  [20,6,6], cell_type=mesh.CellType.hexahedron)
V = fem.VectorFunctionSpace(domain, ("CG", 1))
## Boundary conditions
"""
As we would like to clamp the boundary at $x=0$, we do this by using a marker function, which locate the facets where $x$ is close to zero by machine prescision.
"""
def clamped_boundary(x):
    return np.isclose(x[0], 0)

fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, clamped_boundary)

u_D = np.array([0,0,0], dtype=ScalarType)
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets), V)
# As we want the traction $T$ over the remaining boundary to be $0$, we create a `dolfinx.Constant`
T = fem.Constant(domain, ScalarType((0, 0, 0)))
"""
We also want to specify the integration measure $\mathrm{d}s$, which should be the integral over the boundary of our domain. We do this by using `ufl`, and its built in integration measures
"""
ds = ufl.Measure("ds", domain=domain)
## Variational formulation
#We are now ready to create our variational formulation in close to mathematical syntax, as for the previous problems.
def epsilon(u):
    return ufl.sym(ufl.grad(u)) # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)
def sigma(u):
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2*mu*epsilon(u)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
V_von_mises = fem.FunctionSpace(domain, ("DG", 0))
stresses = fem.Function(V_von_mises)
xdmf = io.XDMFFile(domain.comm, "deformation.xdmf", "w")
xdmf.write_mesh(domain)
for t in np.linspace(0,1, 11):
    f = fem.Constant(domain, ScalarType((0, 0, -rho*g*t)))
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L = ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * ds
    """
    Note that we used `nabla_grad` and optionally `nabla_div` for the variational formulation, as oposed to our previous usage of 
    `div` and `grad`. This is because for scalar functions $\nabla u$ has a clear meaning
    $\nabla u = \left(\frac{\partial u}{\partial x}, \frac{\partial u}{\partial y}, \frac{\partial u}{\partial z} \right)$.

    However, if $u$ is vector valued, the meaning is less clear. Some sources define $\nabla u$ as a matrix with the elements $\frac{\partial u_j}{\partial x_i}$, while other  sources prefer 
    $\frac{\partial u_i}{\partial x_j}$. In DOLFINx `grad(u)` is defined as the amtrix with element $\frac{\partial u_i}{\partial x_j}$. However, as it is common in continuum mechanics to use the other definition, `ufl` supplies us with `nabla_grad` for this purpose.
    ```

    Solve the linear variational problem
    As in the previous demos, we assemble the matrix and right hand side vector and use PETSc to solve our variational problem

    """
    problem = fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()
    ## Visualization

    ## Stress computation
    #As soon as the displacement is computed, we can compute various stress measures. We will compute the von Mises stress defined as $\sigma_m=\sqrt{\frac{3}{2}s:s}$ where $s$ is the deviatoric stress tensor $s(u)=\sigma(u)-\frac{1}{3}\mathrm{tr}(\sigma(u))I$.
    s = sigma(uh) -1./3*ufl.tr(sigma(uh))*ufl.Identity(len(uh))
    von_Mises = ufl.sqrt(3./2*ufl.inner(s, s))
    #The `von_Mises` variable is now an expression that must be projected into an appropriate function space so that we can visualize it. As `uh` is a linear combination of first order piecewise continuous functions, the von Mises stresses will be a cell-wise constant function.
    stress_expr = fem.Expression(von_Mises, V_von_mises.element.interpolation_points())
    stresses.interpolate(stress_expr)

    """
    We could also use Paraview for visualizing this.
    As explained in previous sections, we save the solution with `XDMFFile`.
    After opening the file `deformation.xdmf` in Paraview and pressing `Apply`, one can press the `Warp by vector button` ![Warp by vector](warp_by_vector.png) or go through the top menu (`Filters->Alphabetical->Warp by Vector`) and press `Apply`. We can also change the color of the deformed beam by changing the value in the color menu ![color](color.png) from `Solid Color` to `Deformation`.
    """
    uh.name = "Deformation"
    xdmf.write_function(uh, t)
    xdmf.write_function(stresses, t)
xdmf.close()