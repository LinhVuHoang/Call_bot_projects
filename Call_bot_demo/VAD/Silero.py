from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
model = load_silero_vad()
# audio_path="/home/linhhv/working/lvh/Call_Bot/Call_bot_demo/STT_VN/download.wav"

def get_vad_timestamps(audio_path):
    wav = read_audio(audio_path)
    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        return_seconds=True,  # Return speech timestamps in seconds (default is samples)
        )
    
    return speech_timestamps

