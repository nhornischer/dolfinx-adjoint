# DOLFINx-ADJOINT

An automatic differentiation library for [DOLFINx](https://github.com/FEniCS/dolfinx) that computes gradients using the adjoint method. This package enables efficient sensitivity analysis and gradient-based optimization for finite element simulations.

## Overview

`DOLFINx-ADJOINT` provides automatic differentiation capabilities for DOLFINx by constructing a computational graph during the forward simulation. The library then uses the adjoint method to efficiently compute gradients of scalar objective functions with respect to input parameters, boundary conditions, and other quantities of interest.

### Key Features

- **Automatic differentiation** through computational graph construction
- **Adjoint method** for efficient gradient computation
- **Seamless integration** with DOLFINx workflow
- **Support for complex PDEs** including nonlinear problems
- **Graph visualization** for debugging and understanding dependencies
- **Minimal code changes** required to existing DOLFINx code

### How It Works

The library works by:

1. **Building a computational graph** during the forward simulation that tracks all operations and dependencies
2. **Recording operations** on DOLFINx objects (Functions, Constants, boundary conditions, etc.)
3. **Performing backpropagation** through the graph to compute gradients using the adjoint method

The computational graph is represented as a directed acyclic graph (DAG) where:
- **Nodes** represent DOLFINx objects (Functions, Constants, etc.)
- **Edges** represent operations and their derivatives

This approach enables efficient computation of gradients even for complex PDE systems with many parameters.

## Installation

```bash
git clone https://github.com/nhornischer/dolfinx-adjoint.git
cd dolfinx-adjoint
pip install -e .
```

Requires Python >= 3.12 and [DOLFINx](https://github.com/FEniCS/dolfinx) >= 0.10.0. For demos and tests, use `pip install -e ".[all]"`.

## Demos

The `demos/` directory contains comprehensive Jupyter notebook examples demonstrating various PDE problems:

- **[poisson.ipynb](demos/poisson.ipynb)** - Poisson equation with gradient computation w.r.t. source term, diffusion coefficient, and boundary conditions
- **[heat_equation.ipynb](demos/heat_equation.ipynb)** - Time-dependent heat equation with sensitivity analysis
- **[linear_elasticity.ipynb](demos/linear_elasticity.ipynb)** - Linear elasticity problem with parameter sensitivities
- **[stokes.ipynb](demos/stokes.ipynb)** - Stokes flow with obstacle, computing gradients w.r.t. viscosity and boundary conditions

## Acknowledgments

This library builds upon the excellent work of the [FEniCS Project](https://fenicsproject.org/) and uses [DOLFINx](https://github.com/FEniCS/dolfinx) as its foundation.
