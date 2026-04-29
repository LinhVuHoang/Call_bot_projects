import gradio as gr
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import torch
import librosa
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor


model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "/home/linhhv/working/lvh/Call_Bot/Voice-gender-detection/Wav2vec2_hugging_face",
    local_files_only=True
)

processor = Wav2Vec2FeatureExtractor.from_pretrained(
    "/home/linhhv/working/lvh/Call_Bot/Voice-gender-detection/Wav2vec2_hugging_face",
    local_files_only=True
)

# Label mapping
id2label = {
    "0": "female",
    "1": "male"
}


def classify_audio(audio_path):
    #Load and resample audio to 16kHz
    speech, sample_rate = librosa.load(audio_path, sr=16000)

    #Process audio
    inputs = processor(
        speech,
        sampling_rate = sample_rate,
        return_tensors ="pt",
        padding=True
    )

    with torch.no_grad():
        outputs  = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    
    prediction = {
        id2label[str(i)]: round(probs[i],3) for i in range(len(probs))

    }
    return prediction
# audio_path ="/home/linhhv/working/lvh/Call_Bot/Voice-gender-detection/Data/test_women.wav"
# result = classify_audio(audio_path)
# max_key = max(result, key =result.get) #dùng result.get(key) để lấy value so sánh
# max_value = result[max_key]
# print(max_key,max_value)

def classify_audio_chunk(speech, sample_rate):
    #Load and resample audio to 16kHz
    # speech, sample_rate = librosa.load(audio_path, sr=16000)

    #Process audio
    inputs = processor(
        speech,
        sampling_rate = sample_rate,
        return_tensors ="pt",
        padding=True
    )

    with torch.no_grad():
        outputs  = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    
    prediction = {
        id2label[str(i)]: round(probs[i],3) for i in range(len(probs))

    }
    return prediction