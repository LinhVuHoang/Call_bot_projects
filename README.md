# Vietnamese Voice AI Call Bot

An experimental Vietnamese Voice AI assistant that enables real-time voice interaction through a WebSocket-based communication pipeline.

The system integrates Speech-to-Text (STT), Voice Activity Detection (VAD), Large Language Models (LLMs), and Text-to-Speech (TTS) to process user speech and generate natural Vietnamese voice responses.

> **Project Status:** Experimental / Learning Project

## 1. Overview

This project explores the development of a voice-based conversational AI system designed for call center and virtual assistant scenarios.

Users can communicate with the system through their microphone. Incoming audio is continuously transmitted to the backend, where speech activity is detected and audio segments are processed through an AI pipeline.

The system generates a textual response using an LLM, converts it into speech using a TTS model, and sends the generated audio back to the client.

## 2. System Architecture

```text
                 User / Caller
                       |
                       v
              Browser Microphone
                       |
                       v
                FastAPI WebSocket
                       |
                       v
                  Silero VAD
                       |
                       v
               Audio Buffering
                       |
                       v
                 Async Queue
                       |
                       v
              Background Worker
                       |
                       v
              Audio Processing
                       |
             +---------+---------+
             |                   |
             v                   v
        Noise Reduction      Audio Processing
             |
             v
       Whisper Medium
             |
             v
        Text Normalization
             |
             v
     Voice Gender Detection
             |
             v
        Turn Detection
             |
             v
          OpenAI LLM
             |
             v
          viXTTS TTS
             |
             v
       Generated Audio
             |
             v
        WebSocket Response
             |
             v
          User / Caller
```

*Note: This diagram represents the intended high-level processing architecture. Some modules are present in the project but their exact execution and integration should be verified against their implementations.*

## 3. Technology Stack

| Category | Technologies |
|---|---|
| Backend | Python, FastAPI |
| Communication | WebSocket |
| Asynchronous Processing | asyncio, Async Queue, Background Worker |
| Deep Learning | PyTorch |
| Voice Activity Detection | Silero VAD |
| Speech-to-Text | OpenAI Whisper Medium |
| Noise Reduction | Noisereduce |
| Voice Classification | Audio-based Gender Classification |
| Text Processing | XLM-RoBERTa-based Normalization |
| Turn Detection | Ten Turn Detection |
| Large Language Model | OpenAI API |
| Text-to-Speech | VITS, viXTTS |
| Audio Processing | NumPy, Librosa, Pydub |
| Database | MongoDB / BSON-based audio serialization (verify database implementation) |

## 4. Core Features

### 4.1. Real-time Audio Communication

- Implemented bidirectional audio communication using FastAPI WebSocket.
- Continuously receives microphone audio from the client.
- Sends processing status updates and generated audio responses back to the client.
- Uses a persistent WebSocket connection during a conversation session.

### 4.2. Voice Activity Detection

Integrated Silero VAD to detect speech activity in incoming audio.

Key processing steps:

1. Convert incoming audio bytes into normalized floating-point samples.
2. Divide audio into smaller chunks of 512 samples.
3. Evaluate speech probability using Silero VAD.
4. Detect speech and silence periods using a configurable threshold.
5. Combine audio chunks into complete utterances after a silence interval.

### 4.3. Asynchronous Audio Processing

Implemented an asynchronous processing architecture to separate audio reception from AI inference.

- Uses `asyncio.Queue` to manage incoming audio segments.
- Uses background workers to process queued audio.
- Uses `asyncio.to_thread()` to offload synchronous and computationally intensive operations.
- Helps prevent blocking the WebSocket receiving loop while AI components are processing audio.

### 4.4. Speech-to-Text

Integrated Whisper Medium to convert Vietnamese speech into text.

The recognized transcript is used as the input for subsequent conversational processing.

### 4.5. Audio Preprocessing

Uses Noisereduce to reduce background noise before speech recognition and voice classification.

### 4.6. Voice Analysis

Includes additional speech-related components:

- Voice gender classification to estimate voice characteristics.
- Text-based turn analysis to examine recognized user utterances.
- Text normalization using an XLM-RoBERTa-based module.

### 4.7. LLM-based Conversation

Integrated OpenAI LLM to generate conversational responses.

A custom system prompt is designed for voice-based interactions:

- Natural and polite Vietnamese conversational style.
- Short and clear responses.
- Typically one to three sentences.
- Avoids Markdown and unnecessary special characters.
- Optimizes text formatting for Vietnamese speech synthesis.

### 4.8. Text-to-Speech

Integrated VITS and viXTTS components for speech synthesis.

The current WebSocket pipeline uses viXTTS to convert LLM-generated responses into audio, which is then transmitted to the client.

### 4.9. Conversation Logging

Maintains separate conversation histories for:

- Text transcripts: user input and assistant responses.
- Audio records: original user audio and generated assistant audio.

Audio data is serialized using BSON Binary before being passed to the database layer.

## 5. Project Structure

```text
Call_bot_demo/
│
├── agents/
│   └── agent.py
│
├── app/
│   ├── api.py
│   └── uploads/
│
├── db/
│   └── app.py
│
├── demo/
│   └── voice-agent.html
│
├── LLM/
│   └── open_ai_llm.py
│
├── Normalization_data/
│   └── xlm_roberta_capu.py
│
├── Record/
│   └── Record.py
│
├── Record_voice/
│   └── record_voice.py
│
├── STT_VN/
│   └── Whisper_medium.py
│
├── TTS/
│   ├── Vits.py
│   └── viXTTS_voice.py
│
├── Turn_detection/
│   └── Ten_Turn_Detection.py
│
├── VAD/
│   └── Silero.py
│
└── Voice_gender_detection/
    └── voice_detection.py
```

## 6. Audio Processing Workflow

The main WebSocket endpoint is:

```text
/call/ws/audio
```

The processing workflow consists of the following stages:

1. **Audio Reception:** Receive raw audio bytes from the browser.
2. **Audio Conversion:** Convert Int16 audio samples into normalized Float32 arrays.
3. **Speech Detection:** Apply Silero VAD to identify speech activity.
4. **Utterance Segmentation:** Buffer audio chunks until a silence threshold is reached.
5. **Queue Management:** Push completed audio segments into an asynchronous queue.
6. **Background Processing:** Process queued audio independently of the WebSocket receiving loop.
7. **Speech Recognition:** Convert audio into text using Whisper Medium.
8. **LLM Inference:** Generate a response using OpenAI LLM.
9. **Speech Synthesis:** Convert the response into audio using viXTTS.
10. **Audio Delivery:** Send synthesized audio back through WebSocket.
11. **Conversation Logging:** Save text and audio histories when the session ends.

## 7. Key Engineering Concepts

This project explores several practical AI engineering concepts:

- **Voice AI Pipeline:** Connecting speech recognition, language understanding, and speech synthesis.
- **WebSocket Communication:** Maintaining bidirectional communication between frontend and backend.
- **Asynchronous Programming:** Using asyncio to coordinate concurrent I/O operations.
- **Background Task Processing:** Separating audio reception from AI inference.
- **Audio Signal Processing:** Handling audio normalization, segmentation, and noise reduction.
- **LLM Integration:** Connecting language models to voice-based applications.
- **Multimodal Data Handling:** Managing both audio and textual conversation data.

## 8. Current Limitations

This is an experimental implementation rather than a production-ready voice platform.

Known limitations include:

- Speech recognition and TTS are currently performed on complete utterances rather than token-level or chunk-level streaming.
- The system waits for TTS generation to complete before sending the audio response.
- WebSocket session isolation and queue lifecycle require further improvement for multi-user scenarios.
- Voice activity detection and turn detection can be further optimized.
- Error handling, resource management, and conversation persistence need additional improvements.

## 9. Future Improvements

- Implement streaming Speech-to-Text and chunk-based Text-to-Speech.
- Improve interruption handling and support barge-in during bot responses.
- Implement per-session queues and isolated background workers.
- Optimize inference latency and audio processing performance.
- Improve conversation memory and contextual dialogue management.
- Add monitoring, logging, and performance evaluation.
- Containerize the application and improve deployment workflows.

## 10. Disclaimer

This project was developed for experimentation and learning purposes. It demonstrates the integration of speech processing, LLM-based conversation, and asynchronous backend architecture. It is not intended to be considered a production-ready call center platform.