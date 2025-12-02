from setuptools import setup,find_packages

setup(name="HeteroSymNN",
      version="0.1.5",
      packages=find_packages(),
      python_requires=">=3.9",
      author="Dilosch03",
      install_require=["numpy>=2.2.6","sympy>=1.14.0"],
      extras_require={
        "gpu": ["cupy>=13.6.0"]
        })