from setuptools import setup,find_packages

setup(name="HeteroSymNN",
      version="0.2.0a1",
      packages=find_packages(),
      python_requires=">=3.9",
      author="Dilosch03",
      install_requires=["numpy~=2.2.0","sympy~=1.14.0","platformdirs~=4.5"],
      extras_require={
        "gpu": ["cupy~=13.6.0"]
        })