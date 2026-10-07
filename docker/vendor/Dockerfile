# The vendor-code oracle's base image, which `bellwether sandbox-record` builds: a pinned Python, the packages uv.lock
# pins for bellwether and its `vendor` extra, installed by hash, and bellwether. No model file and no credential is ever
# put in it; each run adds the checkpoint's checked files and the corpus in a run image of its own.
FROM python:3.12-slim@sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f
RUN useradd --uid 10001 --user-group --no-create-home --home-dir /tmp sandbox
COPY requirements.txt /opt/bellwether/requirements.txt
RUN pip install --no-cache-dir --no-deps --require-hashes -r /opt/bellwether/requirements.txt
COPY pyproject.toml uv.lock README.md LICENSE /opt/bellwether/
COPY src /opt/bellwether/src
RUN pip install --no-cache-dir --no-deps /opt/bellwether
