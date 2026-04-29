import os

import torch
from scipy.io.wavfile import write
from huggingface_hub import snapshot_download
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import numpy as np
import io

config = XttsConfig()
config.load_json("/home/linhhv/working/model/config.json")
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
XTTS_MODEL = Xtts.init_from_config(config)
XTTS_MODEL.load_checkpoint(config, checkpoint_dir="/home/linhhv/working/model/")
XTTS_MODEL.eval()
XTTS_MODEL.to(device)

gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
    audio_path="/home/linhhv/working/model/vi_sample.wav",
    gpt_cond_len=XTTS_MODEL.config.gpt_cond_len,
    max_ref_length=XTTS_MODEL.config.max_ref_len,
    sound_norm_refs=XTTS_MODEL.config.sound_norm_refs,
)


# def generalize_audio_from_text(text):
#     print(text)
#     out_wav = XTTS_MODEL.inference(
#     text=text,
#     language="vi",
#     gpt_cond_latent=gpt_cond_latent,
#     speaker_embedding=speaker_embedding, 
#     temperature=0.3,
#     length_penalty=1.0,
#     repetition_penalty=10.0,
#     top_k=30,
#     top_p=0.85, 
# )
#     # lấy audio từ output của XTTS
#     audio_data = np.array(out_wav["wav"], dtype=np.float32)
#     # lưu ra file wav
#     write("voice_cloning.wav", 24000, (audio_data * 32767).astype(np.int16))
    

def generalize_tts_original_text(text):
    out_wav = XTTS_MODEL.inference(
    text=text,
    language="vi",
    gpt_cond_latent=gpt_cond_latent,
    speaker_embedding=speaker_embedding, 
    temperature=0.3,
    length_penalty=1.0,
    repetition_penalty=10.0,
    top_k=30,
    top_p=0.85, 
)
    # lấy audio từ output của XTTS
    audio_data = np.array(out_wav["wav"], dtype=np.float32)
    # lưu ra file wav
    write("natural_voice_cloning.wav", 24000, (audio_data * 32767).astype(np.int16)) 
    # Chuyển đổi sang Int16 chuẩn WAV
    scaled_audio = (audio_data * 32767).astype(np.int16)
    
    # Tạo buffer trong RAM để giả lập file WAV
    byte_io = io.BytesIO()
    write(byte_io, 24000, scaled_audio) # XTTS thường ra 24kHz
    
    return byte_io.getvalue() # Trả về chuỗi bytes của file WAV
    