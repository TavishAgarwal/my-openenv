FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

ENV PYTHONPATH=/app
ENV PORT=7860

EXPOSE 7860

# Default: serve the API (judges will hit /health and /reset)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
