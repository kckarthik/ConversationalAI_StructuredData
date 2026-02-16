FROM python:3.11-slim

# System deps for matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libfreetype6-dev libpng-dev libjpeg-dev curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY app.py .
COPY templates/ templates/
COPY static/ static/

# Create runtime dirs
RUN mkdir -p /app/uploads /app/data /app/static/charts

# Non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:5000/api/health || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--threads", "4", \
     "--timeout", "300", "--access-logfile", "-", "app:app"]
