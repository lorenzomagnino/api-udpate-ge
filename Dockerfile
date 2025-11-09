# Use Python 3.11 slim image for a smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (for scientific stack)
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them first (to utilize Docker's caching)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files (including all .py, configs, etc)
COPY . ./

# Show logs in real-time
ENV PYTHONUNBUFFERED=1

# Run your main orchestrator script
CMD ["python", "main.py"]
