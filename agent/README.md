## Deploy Llama Serve with vLLM

### Without Docker

```bash
vllm serve /home/tananh/llm_subject/finetuning/models/serve \
  --port 8001 \
  --api-key a \
  --served-model-name llama-finetune \
  --enable-auto-tool-choice \
  --tool-call-parser llama3_json
```

Or, to serve the Meta Llama 3.1-8B-Instruct model:

```bash
vllm serve NousResearch/Meta-Llama-3-8B-Instruct --dtype auto --api-key a --port 8001 --served-model-name llama-base --enable-auto-tool-choice --tool-call-parser llama3_json
```

---

### With Docker

1. **Build the Docker image:**
   ```bash
   docker build -t llama_7b_finetune_serve .
   ```

2. **Run the Docker container:**
   ```bash
   docker run \
     -v /home/tananh/llm_subject/finetuning/models/serve:/weights \
     -p 8001:8001 \
     llama_7b_finetune_serve
   ```

---

## Run the MCP Server

Open a new terminal and run:

```bash
python agent/mcp_server.py
```

---

## Run the Chatbot

Open another terminal and run:

```bash
python agent/chatbot.py
```


## Project Structure

```
llm_subject/
├── agent/
│   ├── chatbot.py
│   ├── mcp_server.py
│   └── README.md
├── finetuning/
│   └── models/
│       └── serve/
└── Dockerfile
```

- **agent/**: Contains the main application scripts.
  - `chatbot.py`: Script to run the chatbot interface.
  - `mcp_server.py`: Script to start the MCP server for model communication.
  - `README.md`: Documentation and usage instructions.
- **finetuning/models/serve/**: Directory for storing the fine-tuned model weights used by the server.
- **Dockerfile**: Used to build the Docker image for serving the model.

This structure separates code, model data, and deployment configuration for clarity and maintainability.