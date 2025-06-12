# NLP_PROJECT: Efficient QA and Chatbot with LLaMA-2, MiniLM, and MCP

This project demonstrates advanced NLP workflows using large language models (LLMs) for Question Answering (QA) and chatbot deployment.  
It covers fine-tuning LLaMA-2 on SQuAD 2.0, distillation to MiniLM for efficient inference, and building a modular chatbot agent with tool integration using the Model Context Protocol (MCP).  
**Local LLM deployment is achieved via [vLLM](https://github.com/vllm-project/vllm) for fast and scalable inference.**

---

## 🚀 Main Features

- **Fine-tune LLaMA-2-7B on SQuAD 2.0**  
  Leverages PEFT (LoRA adapters) and QLoRA quantization for memory-efficient training.

- **Knowledge Distillation to MiniLM**  
  Trains a compact student model with strong QA performance using knowledge distillation from LLaMA-2.

- **Chatbot Agent with MCP Tool Integration**  
  Modular agent routes queries to specialized tools:  
  - Document/context retrieval (Vector DB + RAG)
  - English-Korean translation
  - PubMed API for medical/academic search

- **Local vLLM Deployment**  
  Inference served via vLLM, supporting high-throughput and low-latency QA/chatbot responses.

---

## 🧑‍💻 Group Members

- Shriman Juniad Kakar  
- Nguyen Tan Anh

---

## 📚 Project Structure

```
NLP_PROJECT/
├── data_utils.py         # SQuAD v2 data loading and preprocessing
├── model_utils.py        # MiniLM model and tokenizer utilities
├── metrics.py            # SQuAD metrics: Exact Match, F1
├── train_student.py      # Knowledge distillation: train MiniLM student
├── evaluate.py           # Evaluate model on SQuAD v2 validation
├── inference.py          # Run QA inference on new data
├── plotting.py           # Plot training and evaluation metrics
├── chatbot.py            # Main chatbot agent; async vLLM/MCP requests
├── mcp_server.py         # FastAPI MCP server; tool routing
├── translate_tool.py     # Translation tools (NLLB, Google Translate)
├── pubmed_tool.py        # PubMed API integration
├── large_llm.py          # RoBERTa-based QA pipeline example
├── student/              # MiniLM checkpoints, tokenizer, configs
└── models/               # LLM weights and configs for deployment
```


---

## ⚙️ Key Workflows

### 1. Fine-tuning LLaMA-2-7B (with PEFT + QLoRA)
- LoRA adapters: Efficient, parameter-light fine-tuning.
- QLoRA: 4-bit quantization for memory savings.
- Target: Strong QA model on SQuAD 2.0.

### 2. Distillation to MiniLM
- Script: `train_student.py`
- Knowledge distillation from LLaMA-2 to MiniLM for lightweight, fast inference.
- Checkpoints and tokenizer files saved in `/student`.

### 3. Model Evaluation
- Script: `evaluate.py`
- Metrics: SQuAD v2 (Exact Match, F1), using `metrics.py`.

### 4. Inference on New Data
- Script: `inference.py`
- Loads trained student model for fast, accurate QA.

### 5. Visualization
- Script: `plotting.py`
- Visualize loss, accuracy, and other training metrics.

### 6. Chatbot Agent + MCP Integration
- Agent routes user queries via MCP:
  - Context/document retrieval
  - Translation (English ↔ Korean)
  - PubMed/ArXiv search
- Tools integrated as Python modules or APIs.

### 7. Local vLLM Deployment
- Fast, GPU-backed LLM inference (e.g., LLaMA-2, LLaMA-3).
- Supports high-throughput QA/chatbot scenarios.

---

## 🛠️ Notable Modules

- `data_utils.py`: SQuAD v2 loading/preprocessing.
- `model_utils.py`: MiniLM and tokenizer loading.
- `metrics.py`: SQuAD metrics computation.
- `chatbot.py`: Agent interface; async request handling.
- `mcp_server.py`: FastAPI MCP server; tool routing logic.
- `translate_tool.py`: NLLB (Hugging Face) & Google Translate API.
- `pubmed_tool.py`: PubMed query and summary fetch.
- `large_llm.py`: RoBERTa-based QA pipeline example.

## 📺 Demo

[Demo Video (Google Drive)](https://drive.google.com/file/d/1o5DN_6TuTF4YVhIOMGHVaWtHd4koxtPA/view?usp=sharing)

---

## 📖 References

- [LLaMA-2 (Meta)](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)
- [MiniLM (Microsoft)](https://github.com/microsoft/unilm/tree/master/minilm)
- [vLLM](https://github.com/vllm-project/vllm)
- [PEFT (LoRA)](https://huggingface.co/docs/peft)
- [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/)
- [MCP (Model Context Protocol)](https://modelcontext.org/)

---

## 🤝 Acknowledgements


