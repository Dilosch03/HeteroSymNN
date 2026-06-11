import sys
from unittest.mock import patch, MagicMock

class MockRuntime:
    @staticmethod
    def getDeviceCount():
        return 2
class MockCuda:
    runtime = MockRuntime()
class MockCuPy:
    cuda = MockCuda()
    @staticmethod
    def asnumpy(x): return x

with patch.dict('sys.modules', {'cupy': MockCuPy}):
    try:
        import cupy
        print(cupy.cuda.runtime.getDeviceCount())
    except Exception as e:
        print("EXCEPTION:", repr(e))
