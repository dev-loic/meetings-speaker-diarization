FROM python:3.8.6-buster

COPY msd /msd
COPY api /api
COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN apt-get install libsndfile1

CMD uvicorn api.api:app --host 0.0.0.0 --port $PORT