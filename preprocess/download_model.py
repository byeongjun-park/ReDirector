from modelscope import snapshot_download
from huggingface_hub import hf_hub_download

# Download base model
snapshot_download("PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera", local_dir="models/PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera")

# Download ReDirector
hf_hub_download(repo_id="byeongjun-park/ReDirector", filename="step20000.safetensors", local_dir=f'models')