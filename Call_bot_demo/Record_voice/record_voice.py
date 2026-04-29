import sounddevice as sd
import soundfile as sf

duration = 12
sample_rate = 16000

audio = sd.rec(int(duration * sample_rate),
               samplerate=sample_rate,
               channels=1)

sd.wait()

sf.write("test.wav", audio, sample_rate)