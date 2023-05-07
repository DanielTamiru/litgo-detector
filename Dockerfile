FROM python:3.10.11-buster

WORKDIR /app

COPY ./requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . /app

WORKDIR /app/app

ENTRYPOINT [ "python3" ]

CMD ["serve.py"]

