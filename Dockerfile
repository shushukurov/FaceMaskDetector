FROM python:3.7

WORKDIR /app

ADD . /app

RUN apt update \
    && apt install -y htop python3-dev wget

RUN pip install -r requirements.txt \
    &&  apt-get install ffmpeg libsm6 libxext6  -y 

EXPOSE 5000

CMD [ "python","app.py" ]