# syntax=docker/dockerfile:1
FROM python:3.13-slim

LABEL maintainer="Elias Oulad Brahim <obe@cloudhabil.com>"
LABEL organization="Cloudhabil"
LABEL description="Pharmacological Lattice Quantization (PLQ) v2.5 with ToposLang, 3D DACC, and Allostery"

WORKDIR /app

# Ensure non-buffered python output
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH="/app/PLQ:/app/PLQ/topos-lang/src:${PYTHONPATH}"

# Copy PLQ package and topos-lang
COPY plq/ /app/PLQ/plq/
COPY tests/ /app/PLQ/tests/
COPY topos-lang/ /app/PLQ/topos-lang/
COPY README.md /app/PLQ/README.md
COPY pyproject.toml /app/PLQ/pyproject.toml

# Install PLQ package in editable mode
RUN pip install --no-cache-dir -e /app/PLQ

# Verify unit test suite during build
RUN python3 -m unittest discover /app/PLQ/tests -v

WORKDIR /workspace

# Default entrypoint runs PLQ CLI
ENTRYPOINT ["python3", "-m", "plq.cli"]
CMD ["run-phases"]
