FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py ./
COPY ga3 ./ga3
ENV PORT=7860 GA3_DATA_DIR=/tmp
CMD ["sh","-c","uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]
