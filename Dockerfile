# Dockerfile untuk Fundamental ML/DL
FROM python:3.11-slim

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system-level dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    git \
    curl \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip dan install pip tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy only requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files (optional check if folder exists)
COPY projects/ /app/projects/
COPY notebooks/ /app/notebooks/
COPY data/ /app/data/
COPY logs/ /app/logs/

# Buat direktori tambahan jika dibutuhkan
RUN mkdir -p /app/projects/fundamental_ml \
    /app/projects/llm_project \
    /app/data \
    /app/logs

# Expose port JupyterLab
EXPOSE 8080

# Default: Jalankan JupyterLab tanpa token dan password
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8080", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
