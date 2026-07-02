"""Test bootstrap: put the project root on sys.path so `import proxy_server`
resolves regardless of the working directory the tests run from.

Loaded automatically by both `unittest discover` (as a module in the tests
package dir) and pytest (as a conftest plugin).
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
