# My contributions: 
- fix: added third client patch to the six MCP integration tests
- fix: conftest now loads .env — two pre-existing e2e tests had been
  silently skipping on every run, both now run and pass
- impl: QR code MCP server (mcp_servers/qr/) — payload builders for
  wifi/url/vcard/text
- impl: registry + config integration for the QR server
- test: unit tests for payloads and server;
- found: README claims a receipt-image e2e test that doesn't exist
- found: numpy<2 pin blocks local installs on Python 3.13

# Team discussions and decisions:
-Reviewed MCP and Long term memory system and merging with the team.