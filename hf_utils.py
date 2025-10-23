# hf_utils.py
from huggingface_hub import HfApi
from config import HF_TOKEN

def whoami():
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN not set in environment.")
    api = HfApi()
    return api.whoami(token=HF_TOKEN)

if __name__ == "__main__":
    info = whoami()
    print(info)
