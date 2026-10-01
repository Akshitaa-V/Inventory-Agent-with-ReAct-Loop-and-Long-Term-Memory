# My contributions:
- Fixed a package-import bug (flat imports incompatible with `python -m agent.main` execution) across main.py, llm_client.py, and loop.py.
- Diagnosed and resolved a performance issue: identified InnKube server-side latency (44-94s, high variance) as the bottleneck, switched the default model to gemma4-31b-it, and added a read_many batch-reading tool to reduce API round trips, bringing runtime down to a consistent 12-15 seconds.
- Fixed a brittle E2E test assertion that failed on price formatting differences (comma vs. no comma) despite correct extraction; made it tolerant of numeric formatting.
- Corrected the README's setup and run instructions (config file location, local run command) to match the current package structure.
- Reworked the presentation slides. 

# Team:
- Discussed the presentation content and structure as a team and resolved several open issues together.
