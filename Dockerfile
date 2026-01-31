# Backend Dockerfile: builds an image for both classification.py and chat_api.py
FROM python:3.11-slim

# Avoid Python buffering
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (optional but often useful)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend code (including fine-tuned model dir)
COPY . .

# Default command (will be overridden by docker-compose)
CMD ["uvicorn", "chat_api:app", "--host", "0.0.0.0", "--port", "9001"]