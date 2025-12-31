```md
# TinyLora – Fine-Tuned TinyLlama with LoRA

This repository contains a TinyLlama model fine-tuned using **LoRA (Low-Rank Adaptation)** and optimized for **lightweight inference**.

You can run this model in **two ways**:
1. Python + Hugging Face (LoRA adapter)
2. Python + GGUF (quantized, llama.cpp style)

---

## 🧠 Model Overview

- Base Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- Fine-Tuning Method: LoRA (PEFT)
- Quantization: Q4_K_M (GGUF)
- Use Cases:
  - Chatbots
  - Educational assistants
  - Lightweight on-prem LLMs
  - Ollama / llama.cpp inference

---

## 📂 Repository Structure

```

.
├── README.md
├── inference_lora.py          # Run using HF + LoRA adapter
├── inference_gguf.py          # Run GGUF model using Python
├── requirements.txt
└── examples/
└── sample_prompts.txt

````

---

## ⚙️ Environment Setup

### 1️⃣ Create Virtual Environment
```bash
python3 -m venv lora-env
source lora-env/bin/activate
````

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 OPTION 1: Run Model Using Hugging Face + LoRA (Python)

This method:

* Loads the **base model**
* Attaches **LoRA adapter**
* Best for research & further fine-tuning

### 🔹 File: `inference_lora.py`

```bash
python inference_lora.py
```

---

## 🚀 OPTION 2: Run Quantized GGUF Model (Python)

This method:

* Uses **merged + quantized GGUF**
* Fastest
* Lowest memory usage
* Best for production

### 🔹 File: `inference_gguf.py`

```bash
python inference_gguf.py
```

---

## 🧪 Example Prompt

```
Explain Kubernetes in simple terms.
```

---

## 🖥️ Hardware Requirements

| Mode      | RAM   | GPU          |
| --------- | ----- | ------------ |
| HF + LoRA | 16 GB | Optional     |
| GGUF Q4   | 8 GB  | Not required |

---

## 📌 Notes

* Ollama **cannot pull LoRA adapters directly**
* GGUF is the recommended format for deployment
* LoRA adapters are best for continued training

---

## 📜 License

Apache-2.0 (same as base model)

---

## 🙌 Credits

* TinyLlama team
* Hugging Face
* llama.cpp
* PEFT / TRL

````

---

# 📦 `requirements.txt`

```txt
torch
transformers
peft
accelerate
bitsandbytes
sentencepiece
huggingface_hub
llama-cpp-python
````

---

# 🧠 Python Inference – LoRA Adapter

## `inference_lora.py`

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_REPO = "NubeeraTech/tinylora-lora"   # Hugging Face repo

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LORA_REPO)
model.eval()

while True:
    prompt = input("\nYou: ")
    if prompt.lower() in ["exit", "quit"]:
        break

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True
    )

    print("\nModel:", tokenizer.decode(output[0], skip_special_tokens=True))
```

---

# ⚡ Python Inference – GGUF (Quantized)

## `inference_gguf.py`

```python
from llama_cpp import Llama

MODEL_PATH = "./tinylora-q4_k_m.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=8
)

print("GGUF model loaded.")

while True:
    prompt = input("\nYou: ")
    if prompt.lower() in ["exit", "quit"]:
        break

    response = llm(
        prompt,
        max_tokens=200,
        temperature=0.7,
        stop=["</s>"]
    )

    print("\nModel:", response["choices"][0]["text"])
```

---

# 🧪 `examples/sample_prompts.txt`

```txt
Explain Kubernetes in simple terms.
What is Docker?
Difference between VM and container.
What is CI/CD?
```

---

## ✅ What This Gives You

✔ Clean README for GitHub / HF
✔ Python-only inference (no Ollama required)
✔ LoRA + GGUF both supported
✔ Beginner-friendly + production-ready
