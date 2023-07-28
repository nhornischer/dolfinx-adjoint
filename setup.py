from setuptools import setup

setup(name='dolfinx_adjoint',
      version='0.1',
      description='Adjoint and automatic differentiation for FEniCSx',
      author="Niklas Hornischer",
      author_email="nh605@cam.ac.uk",
      packages=['dolfinx_adjoint'],
      package_dir={'dolfinx_adjoint': 'dolfinx_adjoint'},
      )