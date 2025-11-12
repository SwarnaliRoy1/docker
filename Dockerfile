# 1. Use official Python base image
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /app

# 3. Install dependencies
COPY req.txt .
RUN pip install --no-cache-dir -r req.txt

# 4. Copy files
COPY . .

# 5. Expose port
EXPOSE 8200

# 6. Command to run the server
CMD ["uvicorn", "iris_fastapi:app", "--host", "0.0.0.0", "--port", "8200"]
