# ─────────────────────────────────────────────────────────────
#  mlops-pipeline — SageMaker Processing Container (Train/Eval)
#  Python 3.12 — synced with pyproject.toml
#
#  Build:  docker build -f docker/TrainEval.dockerfile -t mlops-pipeline .
#  Push:   (tag + push to ECR)
# ─────────────────────────────────────────────────────────────
FROM python:3.12-slim


RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /opt/ml/code

# Copy local utils library
COPY src/utils/ ./utils/

# Install dependencies
COPY DockerImages/requirements-train.txt .
RUN pip install --no-cache-dir -r requirements-train.txt

# Ensure utils is in PYTHONPATH
ENV PYTHONPATH="/opt/ml/code:${PYTHONPATH}"

ENTRYPOINT ["python3"]
