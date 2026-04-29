from scipy.io.wavfile import write
import torch
import librosa
import soundfile as sf
from transformers import pipeline, set_seed

set_seed(555)

# Khởi tạo pipe (giữ nguyên của bạn)
pipe = pipeline(
    task="text-to-speech",
    model="facebook/mms-tts-vie",
    torch_dtype=torch.float16,
    device=0
)

def text_to_speech(input_text, output_file="response_female.wav"):
    # 1. Tạo âm thanh gốc (Giọng nam)
    speech = pipe(input_text)
    audio = speech["audio"].squeeze()
    sampling_rate = speech["sampling_rate"]

    # Chuyển sang float32 để librosa xử lý chính xác
    audio = audio.astype(float)

    # 2. Xử lý thành giọng nữ
    # n_steps=4: Tăng khoảng 4 bán âm (tăng pitch)
    # audio_pitched = librosa.effects.pitch_shift(audio, sr=sampling_rate, n_steps=4)
    
    # # rate=1.1: Tăng tốc độ nói lên 10% (giọng nữ thường nhanh và thanh thoát hơn)
    # audio_fast = librosa.effects.time_stretch(audio_pitched, rate=1.1)
    # Save as WAV file
    
    # 3. Lưu file bằng soundfile (an toàn hơn scipy cho mảng float)
    sf.write(output_file, audio, sampling_rate)
    print(f"Đã lưu giọng nữ vào: {output_file}")
