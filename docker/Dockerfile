#FROM ubuntu:latest
FROM python:3.8.1
COPY ./*.py /exp/
COPY ./requirement.txt /exp/requirement.txt
COPY ./api/app.py /exp/app.py
COPY ./svm_gamma=0.001_C=0.5.joblib /exp/./svm_gamma=0.001_C=0.5.joblib
#COPY ./mod.joblib /exp/./mod.joblib
RUN pip3 install --no-cache-dir -r /exp/requirement.txt
WORKDIR /exp
EXPOSE 5000
CMD ["python3", "./app.py"]