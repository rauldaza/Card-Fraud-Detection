# ─────────────────────────────────────────────────────────────
#  fraud-processing — SageMaker Processing Container (Pretrain)
#  Python 3.12 — handles SplitData + TabPreprocessing steps
#
#  Build:  docker build -f docker/SplitData.dockerfile -t fraud-processing .
#  Push:   (tag + push to ECR)
# ─────────────────────────────────────────────────────────────
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ml/code

# Copy local utils library (needed by tab_preprocessing.py)
COPY src/utils/ ./utils/

# Install dependencies
COPY DockerImages/requirements-pretrain.txt .
RUN pip install --no-cache-dir -r requirements-pretrain.txt

# Ensure utils is in PYTHONPATH
ENV PYTHONPATH="/opt/ml/code:${PYTHONPATH}"

ENTRYPOINT ["python3"]