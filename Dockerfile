# 1. Using the slim Python 3.12.9 image
FROM python:3.12.9-slim

# 2. Prevent Python from writing .pyc files and ensure logs are sent straight to terminal
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# 3. Setup user early
RUN useradd -m fastapiuser
WORKDIR /app
RUN mkdir -p /app/artifacts && \
	chown fastapiuser:fastapiuser /app/artifacts && \
	chmod 777 /app/artifacts

# 4. Install dependencies (using the cache mount from above)
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# 5. Copy files AND change ownership in ONE step
# This prevents a massive, slow "chown" layer later
COPY --chown=fastapiuser:fastapiuser . .

USER fastapiuser

# 6. Expose the port FastAPI usually runs on
EXPOSE 8080

# The shell form allows the container to pick up the $PORT variable from GCP
# The PORT is handled in fastapibackend.py
CMD ["python", "fastapibackend.py"]

