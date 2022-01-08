FROM pytorch/pytorch:latest
ADD . /server
WORKDIR /server
EXPOSE 5000
RUN pip install -r requirements.txt
CMD ["sh", "run.sh"]
