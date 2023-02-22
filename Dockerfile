FROM python:3.9
COPY ./ ./src
WORKDIR ./src
RUN pip install --upgrade pip
RUN pip install -r requirements-predeploy.txt
EXPOSE 3010
ENTRYPOINT ["bash", "./entrypoint.sh" ]
