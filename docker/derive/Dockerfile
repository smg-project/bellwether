# The container `bellwether derive --write` runs in: the encoders the derived image classes were made with, pinned.
# Pillow 12.3.0 (the engine stack's version) and numpy 2.3.5 by wheel hash, libjpeg-turbo's TurboJPEG library for
# the YCCK variants Pillow cannot write, on python:3.12-slim by digest. Built by hand, as docker/tau2 is:
#   podman build -t bellwether-derive docker/derive
#   podman run --rm -v "$PWD:/work" -w /work bellwether-derive python -I -c \
#     'import sys; sys.path.insert(0, "src"); from bellwether.cli import main; sys.exit(main(["derive", "--write"]))'
FROM docker.io/library/python:3.12-slim@sha256:1148dbb014615300fa7e713b2f94f550c0d1c69780c0805f84f093ceee34dd0b
RUN apt-get update && apt-get install -y --no-install-recommends libturbojpeg0=1:2.1.5-4 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --require-hashes -r /tmp/requirements.txt && rm /tmp/requirements.txt
