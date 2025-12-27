FROM python:3.10-slim

# Prevent Python buffering
ENV PYTHONUNBUFFERED=1

# Set workdir
WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY src ./src
COPY static ./static
COPY templates ./templates

# Expose port
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "src.inference.app:app", "--host", "0.0.0.0", "--port", "8000"]