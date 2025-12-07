import sys
import os

sys.path.insert(0, r"C:\Users\dilos\Documents\GitHub\HeteroSymNN")

import HeteroSymNN

compiler_test = HeteroSymNN.JIT.compiler.SymbolicJITCompiler([("a*sigmoid+relu*z",{"a":1.12})],"CPU_PYTHON",0)