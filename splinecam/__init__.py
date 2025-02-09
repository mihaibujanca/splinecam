try:
    import graph_tool  # noqa: F401
except ImportError:
    raise ImportError(
        "The 'graph-tool' package is required but not installed. "
        "Please install it via your system package manager (e.g., on Ubuntu, run: "
        "sudo apt-get install python3-graph-tool) or via conda-forge."
    )
from . import graph
from . import plot
from . import wrappers
from . import utils
from . import models
from . import compute
