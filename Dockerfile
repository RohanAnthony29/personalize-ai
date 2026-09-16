FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src
WORKDIR /app
COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.9.0+cpu
COPY pyproject.toml README.md ./
COPY src ./src
COPY artifacts ./artifacts
EXPOSE 8000
CMD ["uvicorn", "personalize_ai.main:app", "--host", "0.0.0.0", "--port", "8000"]
