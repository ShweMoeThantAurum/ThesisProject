FROM python:3.11-slim

# ============================
# system deps
# ============================
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ============================
# Create non-root user
# ============================
RUN useradd -m aefluser
USER aefluser

# ============================
# PyTorch CPU (small)
# ============================
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# ============================
# Project requirements
# ============================
COPY --chown=aefluser:aefluser requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ============================
# AWS IoT SDK
# ============================
RUN pip install --no-cache-dir awscrt awsiotsdk

# ============================
# Copy full project
# ============================
COPY --chown=aefluser:aefluser . .

CMD ["bash"]
