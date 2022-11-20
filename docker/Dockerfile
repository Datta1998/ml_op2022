#FROM ubuntu:latest
FROM python:3.8.1
COPY ./*.py /exp/
COPY ./requirement.txt /exp/requirement.txt
COPY ./question1_arg.py /question1_arg.py
RUN pip3 install --no-cache-dir -r /exp/requirement.txt
WORKDIR /exp
#CMD ["python3", "./app.py"]