# Base image
FROM python:3.9-slim

# install python 
RUN apt update && \
apt install --no-install-recommends -y build-essential gcc && \
apt clean && rm -rf /var/lib/apt/lists/*

# Copy files to docker
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/

# Install requirements
WORKDIR /
RUN pip3 install -r requirements.txt --no-cache-dir

# Define entrypoint for docker image
# -u: redirects any print statements to our consol
# Otherwise find print statements in docker log
ENTRYPOINT ["python3", "-u", "src/models/train_model.py"]





