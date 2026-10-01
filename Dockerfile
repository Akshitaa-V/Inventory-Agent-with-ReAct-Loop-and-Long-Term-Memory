# Single-stage build.
#
# An earlier version had a "builder" stage that copied exactly what the runtime
# stage copies, installed the same packages, and produced no artifact the
# runtime stage ever consumed -- dead weight, not a multi-stage build. Its
# comments justified it by a test gate on line 63 of the README that was
# removed in dc07cf3, so nothing has run tests at build time since.

FROM python:3.12-slim-bookworm

WORKDIR /app

# System dependencies for OCR: the Tesseract *binary* and language packs are
# separate from the pytesseract Python wrapper installed via pip below --
# pip alone does not provide these. poppler-utils is only needed for
# scanned-PDF OCR (pdf2image shells out to pdftoppm). ocr_server.py calls
# sys.exit(1) at import time if tesseract is missing, so this must be in the
# image that actually runs the code.
#
# gosu is what lets docker/entrypoint.sh repair directory ownership as root
# and then hand over to the non-root `agent` user before anything of ours
# executes. The application itself never runs as root.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-deu \
    poppler-utils \
    gosu \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN useradd --create-home --shell /bin/bash agent

COPY agent/ agent/
COPY mcp_servers/ mcp_servers/
COPY config.json .
COPY samples/receipts/ workspace/receipts/
COPY samples/ samples/
COPY dashboard.py .

# Normalise line endings BEFORE installing the script. git with
# core.autocrlf=true (the Windows default) checks .sh files out with CRLF, and
# a shebang ending in \r makes execve fail with "bad interpreter" before a
# single line of our code runs -- an error that points at the image instead of
# at the checkout. .gitattributes already pins *.sh to LF; this covers anyone
# who edits the file outside git.
COPY docker/entrypoint.sh /tmp/entrypoint.sh
RUN tr -d '\r' < /tmp/entrypoint.sh > /usr/local/bin/entrypoint.sh \
    && chmod +x /usr/local/bin/entrypoint.sh \
    && rm /tmp/entrypoint.sh

# Create /app/data (ChromaDB's storage dir) BEFORE chown, so it is already
# owned by "agent" when the image is built. docker-compose.yml mounts a named
# volume here; Docker seeds a new volume's initial content from whatever is
# already in the image at that path. Without this directory existing first,
# Docker creates the volume fresh as root, the container runs as the non-root
# `agent` user, and ChromaDB fails with "Permission denied (os error 13)".
#
# This only fixes the *first* seed. A volume created by an older image is
# never re-seeded, which is why entrypoint.sh repeats the check at startup.
RUN mkdir -p /app/data && chown -R agent:agent /app

# HOME must be explicit. entrypoint.sh drops to `agent` via gosu, which does
# not modify the environment, so without this the ChromaDB embedding model
# baked in below would be looked for under /root at runtime and re-downloaded
# (or fail outright with no network).
ENV HOME=/home/agent
USER agent

# Pre-download the ChromaDB embedding model so it is baked into the image
RUN python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction; DefaultEmbeddingFunction()('warmup')"

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# The entrypoint starts as root only so it can repair ownership on the two
# state directories; it drops to `agent` with gosu before exec'ing anything.
USER root
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python", "-m", "agent.main"]
