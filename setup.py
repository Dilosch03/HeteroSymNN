from setuptools import setup,find_packages
import pathlib


here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

setup(name="HeteroSymNN",
      version="0.3.0rc4",
      packages=find_packages(),
      python_requires=">=3.10",
      author="Dilosch03",
      entry_points={
          "console_scripts": [
              "heterosymnn=HeteroSymNN.__main__:main",
          ],
      },
      install_requires=["numpy>=2.0","sympy~=1.14","platformdirs~=4.5"],
      extras_require={
        "cuda11": ["cupy-cuda11x>=12.0.0"],
        "cuda12": ["cupy-cuda12x>=12.0.0"],
        "cuda13": ["cupy-cuda13x>=12.0.0"]
        },
      description="Framework for Heterogeneous Neural Networks using Symbolic Mathematics and Automatic Differentiation using in runtime compilation to generate code for the GPU or paralel CPU.",
      long_description=long_description,
      long_description_content_type="text/markdown",
      project_urls={
        "Documentation": "https://heterosymnn.readthedocs.io/",
        "Source": "https://github.com/Dilosch03/HeteroSymNN",
        "Tracker": "https://github.com/Dilosch03/HeteroSymNN/issues",
    })