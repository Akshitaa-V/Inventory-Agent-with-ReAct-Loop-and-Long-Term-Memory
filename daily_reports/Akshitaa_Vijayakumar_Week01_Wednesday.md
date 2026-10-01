# My contributions:
- Diagnosed and fixed a configuration bug where the agent could not locate the workspace folder due to a missing workspace_root path, causing tool calls to silently fail.
- Rebuilt the agent's system prompt and wiring to support a general-purpose inventory use case, extracting structured fields (item, category, price, date, vendor, serial number, condition, location) rather than a narrow single-domain schema.
- Verified the full pipeline end-to-end: agent correctly navigated a receipts folder, read multiple files, and generated a structured inventory.md via real tool calls against the InnKube API.
- Added a samples folder with example receipts (electronics, furniture, and medical equipment) so the demo is reproducible from a fresh clone, since the working workspace folder is intentionally excluded from version control.
- Rewrote the project README with an overview, Week 1 status summary, and clear setup/demo instructions for anyone cloning the repository.

# Team:
- Finalized the project topic as a team and began implementation work on the chosen use case.
