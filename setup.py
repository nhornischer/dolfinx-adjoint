from setuptools import setup

extras = {
      'test': ['pytest'],
      'visualization': ['networkx'],
}

extras['all'] = [item for group in extras.values() for item in group]

setup(name='dolfinx_adjoint',
      version='0.2',
      description='Automatic differentiation library for FEniCSx',
      author="Niklas Hornischer",
      author_email="nh605@cam.ac.uk",
      packages=['dolfinx_adjoint'],
      package_dir={'dolfinx_adjoint': 'dolfinx_adjoint'},
      install_requires=['mpi4py', 'numpy', 'fenics-dolfinx>=0.6.0'],
      extras_require=extras,
      )