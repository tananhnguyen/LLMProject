FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip install vllm
WORKDIR /app
EXPOSE 8001
ENTRYPOINT ["vllm", "serve", "/weights", "--port", "8003", "--api-key", "a"]
