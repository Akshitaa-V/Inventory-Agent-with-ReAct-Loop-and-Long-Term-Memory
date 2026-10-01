# My contributions:
- impl: replaced the Google Drive MCP server with a Product Recall MCP tool (wraps the official US CPSC SaferProducts.gov API) — no login required, so results are identical for every tester
- test: added unit tests for the recall tool, covering input validation, formatting, and network failure handling
- fix: found and fixed a case-sensitivity bug in file path resolution — worked fine on Windows, broke on the Docker image's Linux filesystem
- fix: found and fixed a bug where a Docker volume mount could hide the image's built-in sample receipts; workspace now repopulates itself automatically
- impl: added CLI table rendering so markdown tables in agent responses display as aligned grids instead of raw text
- test: added tests for MCP integration mechanics (config-driven enable/disable, failure isolation, call routing) and path resolution
- docs: updated the README (corrected setup steps, added Podman guidance, documented new features)

# Team:
- Talked through possible approaches for the MCP integration and long-term memory.
