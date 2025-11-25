from modelscope import snapshot_download
import gdown

# Download base model
snapshot_download("PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera", local_dir="models/PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera")

# Download ReDirector
gdown.download_folder(id="173XwLyoRhYCiH57Ho_hHkZW40vH_DO7Q",
    output='models',
    quiet=False,
    use_cookies=False,
)