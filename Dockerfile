# Runs on EB (x86_64) and on your Mac via emulation
FROM --platform=linux/amd64 python:3.8-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Minimal system libs for numpy/scipy/opencv/open3d
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    libgl1 libglib2.0-0 libx11-6 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first (cache-friendly)
COPY requirements.txt .
RUN python -m pip install -U pip && pip install -r requirements.txt

# Copy app code
COPY . .

# Container listens on 8000
EXPOSE 8000

# Verbose logs while we verify
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]

