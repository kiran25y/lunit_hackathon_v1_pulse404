FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# CoEval normally connects on 8000; PORT keeps the image compatible with
# runners that inject a different listening port.
CMD ["sh", "-c", "exec python -m uvicorn app.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
