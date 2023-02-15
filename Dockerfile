# software we will be using
FROM python:3.9
# Copying all the source folders to the Docker Environment
COPY ./ ./src
# Specifying Working Directory
WORKDIR ./src
RUN echo "Working Directory"
# Installing requirements using txt file
RUN pip install -r requirements.txt
# Exposing the required port number
EXPOSE 3010
# Final command to start the API
CMD ["python", "./Trading Bot/Trading Model/Stock_price_pred.py"]
