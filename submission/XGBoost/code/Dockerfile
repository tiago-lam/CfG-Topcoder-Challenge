FROM ubuntu:20.04

# Install General Requirements
RUN apt-get update && \
        apt-get install -y --no-install-recommends \
        python3.9 \
        python3-pip \
        python3-numpy \
        python3-pandas \
        python3-sklearn \
		python3-xgboost \
		python3-tensorflow \
        software-properties-common

# Create a /work directory within the container, copy everything from the
# build directory and switch there.
RUN mkdir /work
COPY . /work
WORKDIR /work

# Again, test and train scripts should be executable within the container.
RUN chmod +x test.sh
RUN chmod +x train.sh
