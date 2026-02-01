# Base image (Vertex AI & ML friendly)
FROM python:3.10-slim

# Prevent Python buffering
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy requirements first (layer caching)
COPY requirements-infer.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements-infer.txt

# Copy source code
COPY src/ src/
COPY models/ models/
COPY app.py .

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
