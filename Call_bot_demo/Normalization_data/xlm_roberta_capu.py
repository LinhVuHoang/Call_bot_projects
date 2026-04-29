import os
import sys
from huggingface_hub import snapshot_download

# 1. Define a clean local directory
model_dir = "/home/linhhv/working/xlm_roberta_capu_local"
# # # 2. Download the full repo to that specific folder (no nested cache)
# # # local_dir_use_symlinks=False is key here to avoid the "blobs" error
# # snapshot_download(
# #     repo_id="dragonSwing/xlm-roberta-capu", 
# #     local_dir=model_dir,
# #     local_dir_use_symlinks=False
# # )

# 3. Add this directory to your path
if model_dir not in sys.path:
    sys.path.append(model_dir)

# 4. Import and Init
from gec_model import GecBERTModel

model = GecBERTModel(
    vocab_path=os.path.join(model_dir, "vocabulary"),
    model_paths=model_dir, # Point directly to local dir
    split_chunk=True
)

def normalize_text(text):
    return (model(text))


