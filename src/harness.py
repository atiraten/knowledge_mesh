"""Run Knowledge Mesh from a source checkout without installing the package.

    python src/harness.py serve            # FastAPI app + browser UI on http://127.0.0.1:8000
    python src/harness.py mcp              # MCP server over stdio
    python src/harness.py check-config     # validate config/mesh.yaml
    python src/harness.py --help           # all commands
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from knowledge_mesh.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
