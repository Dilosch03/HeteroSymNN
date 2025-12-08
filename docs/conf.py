import sys
from unittest.mock import MagicMock

class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

# Mock cupy so docs build without needing a GPU
MOCK_MODULES = ['cupy', 'cupy.cuda']
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)