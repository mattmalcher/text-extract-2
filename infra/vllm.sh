# https://docs.vllm.ai/en/latest/serving/deploying_with_docker.html

docker run \
    --rm \
    --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=$(secret-tool lookup api hf_read)" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model bartowski/Meta-Llama-3-8B-Instruct-AWQ \
    --quantization awq \
    --dtype auto \
    --max-model-len 7000 \
    --gpu-memory-utilization 0.80