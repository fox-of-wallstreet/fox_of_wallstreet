# 1. Using the slim Python 3.12.9 image
FROM python:3.12.9-slim

# 2. Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Set the working directory
WORKDIR /app

# 4. Install dependencies
# Make sure 'fastapi' and 'uvicorn' are in your requirements.txt
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your application code
COPY . .

# 6. Expose the port FastAPI usually runs on
EXPOSE 8000

# 7. Run as a non-root user for security
RUN useradd -m fastapiuser
RUN mkdir -m 777 /app/artifacts
RUN chown -R fastapiuser:fastapiuser /app
USER fastapiuser

# 8. Start the app using Uvicorn
# 'main:app' assumes your file is main.py and your FastAPI instance is named 'app'
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


