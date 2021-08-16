FROM tensorflow/tensorflow:2.3.1-gpu

RUN mkdir /app
WORKDIR /app

# COPY DIR
COPY . .

# RUN
RUN apt-get update
RUN apt-get install python3-dev python3-pip python3-venv -y
RUN python3 -m venv --system-site-packages ./venv
RUN source ./venv/bin/activate
RUN pip install --upgrade pip

RUN pip install -r requirements.txt

# Run unit tests
CMD [ "python" , "tests/runner.py"]