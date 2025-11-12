FROM python:3.11-slim

# System deps (git, build tools, etc. if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Copy only requirements first (if you have them)
# If you don't have requirements.txt, skip this and install in one go later
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# Copy full repo
COPY . .

# Install project in editable mode (or just install deps here)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt || true

# For safety, install explicitly common deps your repo uses
RUN pip install --no-cache-dir \
    torch \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    boto3 \
    pyyaml \
    scikit-learn

# Default command: do nothing (we override in ECS task)
CMD ["bash"]
