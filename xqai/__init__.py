"""xqai — Chinese Chess (Xiangqi) learning AI.

Pure-Python / PyTorch side of the project. The C++ rules kernel is exposed
separately as the compiled extension module ``xqai._xqcore`` (see INTERFACES.md
§3); it may not be built yet, so nothing here imports it at module load time.

Public submodules:
- ``xqai.config``   : load configs/default.yaml as a dotted-access object.
- ``xqai.encoding`` : board/move tensor encoding (INTERFACES.md §4).
- ``xqai.network``  : PVNet policy-value network + losses (INTERFACES.md §5).
"""

__all__ = ["config", "encoding", "network"]

__version__ = "0.1.0"
