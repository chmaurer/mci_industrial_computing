FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

# Expose the internal port matching your mcp.run configuration
EXPOSE 8000

CMD ["python", "server.py"]