"""
modal_app.py
============
Deploys the fine-tuned Llama-3.2-3B price estimator as a Modal serverless
GPU endpoint — week8 architecture using Modal 1.x API.

Setup (one-time):
    pip install modal
    modal setup
    modal secret create huggingface HF_TOKEN=hf_xxx

Deploy:
    modal deploy modal_app.py

After deploying, copy the printed endpoint URL into your .env:
    MODAL_ENDPOINT_URL=https://your-workspace--bargainbuddy-pricer-price.modal.run
"""

import re
import modal

BASE_MODEL_ID = "meta-llama/Llama-3.2-3B"
ADAPTER_ID = "sjgod1247/price-2026-03-30_14.29.05-lite"
MODEL_DIR = "/models"

# Persistent volume to cache model weights across container restarts
model_volume = modal.Volume.from_name("bargainbuddy-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.40.0",
        "peft>=0.10.0",
        "accelerate>=0.30.0",
        "huggingface_hub>=0.23.0",
        "fastapi[standard]",
    )
)

app = modal.App("bargainbuddy-pricer", image=image)


@app.cls(
    gpu="T4",
    image=image,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={MODEL_DIR: model_volume},
)
class Pricer:

    @modal.enter()
    def load_model(self):
        """Download (if not cached) and load the model into GPU memory."""
        import os
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        from peft import PeftModel
        from huggingface_hub import snapshot_download

        token = os.environ["HF_TOKEN"]
        base_path = f"{MODEL_DIR}/base"
        adapter_path = f"{MODEL_DIR}/adapter"

        # Download to volume if not already cached
        if not os.path.exists(f"{base_path}/config.json"):
            print(f"Downloading base model {BASE_MODEL_ID}...")
            snapshot_download(BASE_MODEL_ID, local_dir=base_path, token=token)
            model_volume.commit()

        if not os.path.exists(f"{adapter_path}/adapter_config.json"):
            print(f"Downloading LoRA adapter {ADAPTER_ID}...")
            snapshot_download(ADAPTER_ID, local_dir=adapter_path, token=token)
            model_volume.commit()

        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)

        print("Loading base model in fp16...")
        base = AutoModelForCausalLM.from_pretrained(
            base_path,
            dtype=torch.float16,
            device_map="auto",
        )

        print("Applying LoRA adapter...")
        model = PeftModel.from_pretrained(base, adapter_path)
        model.eval()

        self.pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        print("Model ready.")

    @modal.fastapi_endpoint(method="POST")
    def price(self, item: dict) -> dict:
        """
        Estimate the price of a product.

        Request:   POST {"description": "product text here"}
        Response:  {"price": 49.99}
        """
        description = item.get("description", "")
        prompt = (
            f"How much does this cost to the nearest dollar?\n\n"
            f"{description}\n\n"
            f"Price is $"
        )
        output = self.pipe(prompt, max_new_tokens=10, return_full_text=False)[0]
        text = output["generated_text"].strip().replace("$", "").replace(",", "")
        match = re.search(r"[-+]?\d*\.\d+|\d+", text)
        price_val = float(match.group()) if match else 0.0
        return {"price": price_val}
