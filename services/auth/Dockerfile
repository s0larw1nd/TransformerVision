FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY auth.py .
COPY templates ./templates
COPY static ./static

EXPOSE 82

CMD ["uvicorn", "auth:app", "--host", "0.0.0.0", "--port", "82"]