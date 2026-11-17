
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy dependency file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY src/ ./src/
COPY api/ ./api/

# Copy trained model artifacts
COPY artifacts/ ./artifacts/

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI application
CMD ["python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
