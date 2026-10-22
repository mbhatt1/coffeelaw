# Small-scale test and notebook environment for the ceilings harness (no API keys needed).
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /work
COPY pyproject.toml ./
COPY src ./src
RUN pip install -e ".[dev,notebook]"
# At run time the repository is mounted at /work, so edits apply without a rebuild.
CMD ["pytest", "-q"]
