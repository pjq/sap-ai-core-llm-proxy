"""Test package. Ensures the project root is importable so test modules can
`import proxy_server` under both `unittest discover` and pytest.
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
