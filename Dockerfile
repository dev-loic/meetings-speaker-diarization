FROM python:3.8.6-buster

COPY msd /msd
COPY api /api
COPY requirements.txt /requirements.txt
COPY .env /.env

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y libsndfile1-dev
RUN apt-get update && apt-get install -y ffmpeg

CMD uvicorn api.api:app --host 0.0.0.0 --port $PORT