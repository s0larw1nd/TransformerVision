FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY user.py .
COPY templates ./templates
COPY static ./static

EXPOSE 83

CMD ["uvicorn", "user:app", "--host", "0.0.0.0", "--port", "83"]