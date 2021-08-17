FROM tensorflow/tensorflow:2.3.1-gpu

RUN mkdir /app
WORKDIR /app

# COPY DIR
COPY . .

# Install Deps
RUN apt-get update
RUN apt-get install python3-dev python3-pip -y

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Run the tests
# CMD [ "python", "tests/runner.py" ]
CMD ["coverage", "run", "tests/runner.py", "&&", "coverage", "html" ,"--omit=*/venv/*,*/usr/*,*/lib/*,*/tests/*", "-i"]