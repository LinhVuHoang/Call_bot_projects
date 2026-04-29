from fastapi import FastAPI, UploadFile, File, status, WebSocket, WebSocketDisconnect
import torch
import numpy as np
import os, sys
import asyncio
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))
import uuid
from pydub import AudioSegment
from silero_vad import load_silero_vad
from agents.agent import response_audio_streaming
from TTS.viXTTS_voice import generalize_tts_original_text
from db.app import save_audio_to_log,save_to_log
import time
from datetime import datetime
import uuid
import secrets
import bson
# Cấu hình thiết bị
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_silero_vad().to(device)
model.eval() # Chuyển sang chế độ inference

app = FastAPI(title="Call bot", description="Call bot agent")

#create a queue

audio_queue = asyncio.Queue()

async def worker(websocket,call_history,audio_history):
    """Hàm này chạy ngầm, chuyên lấy âm thanh từ Queue ra xử lý lần lượt"""
    while True:
        #Lấy âm thanh ra (đợi nếu giỏ trống)
        audio_data = await audio_queue.get() # nếu trống vào trạng thái ngủ đông (tạm dừng thực thi), trong lúc ngủ không chiếm dụng CPU, các tác vụ khác hoạt động bình thường
        #await cho phép Hàm khác (Request khác) vượt lên chạy trước, nhưng cấm các dòng code phía dưới nó (trong cùng hàm) chạy tiếp cho đến khi có kết quả.
        try:
            
            #gọi hàm xử lý Ai
            
            await handle_ai_response(websocket,audio_data,call_history,audio_history)

        finally:
            #báo xong mẩu
            audio_queue.task_done()
            
            
            
async def handle_ai_response(websocket, audio_data, call_history,audio_history):
    try:
        
        #Báo cho front-end dừng gửi mic
        await websocket.send_json({"type": "status", "value": "processing"})
        # 1. Gọi LLM/STT
        response_text, audio_duration, raw_text = await response_audio_streaming(audio_data)
        if response_text:
            
            # 2. Chuyển thành âm thanh (TTS)
            audio_bytes = await asyncio.to_thread(generalize_tts_original_text, response_text)
            await websocket.send_json({"type": "status", "value": "listening"})
            chat_template = [
                {"role":"user", "request":raw_text},
                {"role":"assistant", "response": response_text},            
            ]
            audio_template =[
                {"role":"user", "request":audio_data},
                {"role":"assistant", "response": audio_bytes},    
            ]
            
            for turn in audio_template:
                for key in ["request", "response"]:
                    if key in turn and isinstance(turn[key], np.ndarray):
                        # Chuyển numpy -> raw bytes -> BSON Binary
                        turn[key] = bson.Binary(turn[key].tobytes())
            call_history.append(chat_template)
            audio_history.append(audio_template)
            # 3. Gửi lại cho Client
            # Kiểm tra xem socket còn mở không trước khi gửi
            await websocket.send_bytes(audio_bytes)
            print(f"🔊 Đã gửi phản hồi: {response_text}")
            
    except Exception as e:
        print(f"❌ Lỗi xử lý AI ngầm: {e}")


@app.websocket("/call/ws/audio")
async def websocket_endpoint(websocket: WebSocket): #special route to get pipeline to connect continuesly between client and server

    await websocket.accept() #connect between websocket and server
    audio_buffer = [] 
    is_speaking = False
    silence_chunks = 0
    call_history =[]
    audio_history = []
    vad_state = torch.zeros(0).to(device)
    worker_task = asyncio.create_task(worker(websocket, call_history,audio_history))
    start_time = datetime.now()
    
    session_id = f"{int(time.time())}-{secrets.token_hex(4)}"
    try:
        while True: #infinity loop to receive data from user 
                # Nhận dữ liệu với timeout ngắn để không bị treo
                data = await websocket.receive_bytes() #receive voice data
                
                samples = np.frombuffer(data, dtype=np.int16) #conver raw byte to int.16
                audio_fp32 = samples.astype(np.float32) / 32768.0 #normalize data in range [-1:1]
                
                chunk_size = 512
                for i in range(0, len(audio_fp32), chunk_size):
                    chunk = audio_fp32[i : i + chunk_size]
                    if len(chunk) < chunk_size: continue 
                    
                    chunk_tensor = torch.from_numpy(chunk).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        result = model(chunk_tensor, 16000)
                        out, vad_state = result if isinstance(result, tuple) else (result, None)
                        prob = out.item()
                    
                    if prob > 0.7: #vad
                        print("Broswer mic level: ",{prob})
                        if not is_speaking:
                            is_speaking = True
                        silence_chunks = 0
                        audio_buffer.append(chunk)
                    else:
                        if is_speaking:
                
                            silence_chunks += 1
                            audio_buffer.append(chunk) 
                            
                            if silence_chunks > 40:
                                print("🎯 NGỪNG NÓI! Đang xử lý...")
                                full_audio_np = np.concatenate(audio_buffer) #combine chunks to a 
                                
                                # Tách logic xử lý ra để không làm nghẽn việc nhận bytes tiếp theo
                                # QUAN TRỌNG: Không dùng await ở đây!
                                # Tạo một task chạy nền để vòng while được rảnh tay ngay
                                # asyncio.create_task(handle_ai_response(websocket, full_audio_np))
                                await audio_queue.put(full_audio_np)
                               
                                # RESET trạng thái
                                is_speaking = False  
                                silence_chunks = 0
                                audio_buffer = []
                                vad_state = torch.zeros(0).to(device)
                                # await asyncio.sleep(0.3) #ngủ async nên phải có await trong khi time sleep là đóng băng toàn bộ server
                                print("Ready for next sentence...")
                                
            
    except WebSocketDisconnect:
                print("🔌 Client đã ngắt kết nối chủ động")
                # break
    except Exception as e:
        print(f"⚠️ Lỗi trong vòng lặp: {e}")
        # Nếu lỗi quá nặng thì break, nếu không thì continue
        # if "1001" in str(e) or "closed" in str(e).lower():
        #     break
        # continue
    finally:
        # CHỐT CHẶN: Đợi Worker xử lý nốt những gì còn trong Queue
        if not audio_queue.empty():
            print("⏳ Waiting for final tasks in queue...")
            await audio_queue.join()
        end_time= datetime.now()
        if len(call_history)>0:
            
            db_record ={
                "session_id":session_id,
                "transcript":call_history,
                "start_time": start_time,
                "end_time":end_time
            }
            save_to_log(db_record)
        if len(audio_history)>0:
            db_record ={
                "session_id":session_id,
                "transcript":audio_history,
                "start_time": start_time,
                "end_time":end_time
            }
            save_audio_to_log(db_record)
        print(f"💾 Session {session_id} saved to DB.")
        worker_task.cancel() 
        print("🧹 Đã dọn dẹp worker")

# ---  API Upload  ---
@app.post("/call/agents/runs", status_code=status.HTTP_200_OK, tags=["Call"])
async def call_api(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    file_id = str(uuid.uuid4())
    os.makedirs("uploads", exist_ok=True)
    
    raw_path = f"uploads/{file_id}_raw"
    wav_path = f"uploads/{file_id}.wav"
    
    with open(raw_path, "wb") as f:
        f.write(audio_bytes)
        
    audio = AudioSegment.from_file(raw_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(wav_path, format="wav")
    os.remove(raw_path)
    
    return {"wav_path": wav_path}