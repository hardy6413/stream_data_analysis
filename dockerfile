FROM python:3.8

WORKDIR /app

COPY main.py /app/main.py
COPY config.yaml /app/config.yaml

RUN python -m pip install "dask[complete]"

CMD ["python", "main.py"]