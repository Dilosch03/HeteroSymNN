from setuptools import setup,find_packages

setup(name="HeteroSymNN",
      version="0.2.0",
      packages=find_packages(),
      python_requires=">=3.10",
      author="Dilosch03",
      install_requires=["numpy>=2.0,<2.3","sympy~=1.14","platformdirs~=4.5"],
      extras_require={
        "gpu": ["cupy~=13.6"]
        })