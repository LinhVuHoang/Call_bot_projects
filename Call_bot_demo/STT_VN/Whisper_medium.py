import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
import librosa
import os
import time
local_path="/home/linhhv/working/my_whisper_model"

# processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
# processor.save_pretrained(local_path)
# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")
# model.save_pretrained(local_path)
processor = WhisperProcessor.from_pretrained(local_path,local_files_only=True)
model = WhisperForConditionalGeneration.from_pretrained(local_path,local_files_only=True).to('cuda:1')



# ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
# sample = ds[0]["audio"]
# load audio từ url
forced_decoder_ids = processor.get_decoder_prompt_ids(language="vi", task="transcribe")
# audio url
audio_url = "/home/linhhv/working/natural_voice_cloning.wav"


def STT_whisper(audio_url):
    speech_array, sampling_rate = librosa.load(audio_url, sr=16000)
    inputs = processor(
    speech_array,
    sampling_rate=sampling_rate, #phụ trách việc cho model thời gian của audio vì speech_array là một mảng số * sampling_rate = duration
    return_tensors="pt"
)

    input_features = inputs.input_features.to("cuda:1")

    predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    
    return transcription
start_time = time.time()
print(STT_whisper(audio_url))
end_time = time.time()
print(f"Thời gian xử lý (Inference): {end_time - start_time:.2f}s")
def STT_chunk_whisper(audio,sr):
    inputs = processor(
    audio,
    sampling_rate=sr,
    return_tensors="pt"
)

    input_features = inputs.input_features.to("cuda:1")

    predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
    torch.cuda.empty_cache()
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    
    return transcription



# from transformers import WhisperProcessor, WhisperForConditionalGeneration

# # Tải và lưu vào thư mục local
# model_name = "openai/whisper-medium"
# processor = WhisperProcessor.from_pretrained(model_name)
# model = WhisperForConditionalGeneration.from_pretrained(model_name)

# processor.save_pretrained("./my_whisper_model")
# model.save_pretrained("./my_whisper_model")