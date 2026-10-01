# My contributions:
- test: end-to-end round-trip test for the QR server — the agent generates a
  code, it is decoded with zxing-cpp (a different implementation from the
  encoder) and the payload asserted.
- verify: live run against InnKube — the model selected generate_qr_code on
  its own, wrote the file, and the decoded payload matched the request
- fix: Podman on Linux — missing `data/` bind-mount source, SELinux `:Z`
  labels, and rootless `--userns=keep-id`; container now starts and runs
- docs: README — added the QR server, updated layout/config/test sections,
  and rewrote the Podman section with the verified command
- docs: removed the README claim of an e2e test using real receipt images
  (reported Wednesday, no such test existed)
- merge: merged feature/qr-mcp-server into main
- found: `rich` is imported in agent/loop.py but is not in requirements.txt —
  it only resolves through chromadb's transitive dependency
- prep: reviewed the presentation guidelines for Friday's group presentation

# Team discussions and decisions:
- Merged the QR branch into main after the team reviewed it.
- Discussions/reviews with Akshitaa and Imer for friday presentation.
