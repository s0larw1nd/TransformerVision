FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY orchestrator.py .

EXPOSE 81

CMD ["uvicorn", "orchestrator:app", "--host", "0.0.0.0", "--port", "81"]