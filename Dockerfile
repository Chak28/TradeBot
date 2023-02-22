FROM python:3.9
COPY ./ ./src
WORKDIR ./src
RUN pip install --upgrade pip
RUN pip install -r stock.txt
EXPOSE 3000
CMD ["python", "api.py" ]
