# tau2's own code at the release the importer's prompts come from, to build the agents' tool schemas
# (scripts/tau2_tools.py). Python 3.13 as tau2's .python-version asks, pinned by digest; tau2's locked dependencies
# (pdm.lock, the default group), installed by hash; tau2's source and domain data, unbuilt, on the path.
FROM python:3.13-slim@sha256:bf44cdfcb76cd3b41e879bc058fc37ec5872002ccfde7fcb765e218cde0cd79c
RUN useradd --uid 10001 --user-group --no-create-home --home-dir /tmp sandbox
COPY requirements.txt /opt/tau2/requirements.txt
RUN pip install --no-cache-dir --no-deps --require-hashes -r /opt/tau2/requirements.txt
COPY repo /opt/tau2/repo
COPY tau2_tools.py /opt/tau2/tau2_tools.py
RUN chmod -R a-w /opt/tau2
