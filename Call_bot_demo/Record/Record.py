import pyaudio

#cấu hình Audio

FORMAT = pyaudio.paInt16

CHANNELS = 1

RATE= 16000 # chuẩn cho Whisper và viXTTS2

CHUNK = 1024 #Số khung hình mỗi lần thu (khoảng 64ms)

p = pyaudio.PyAudio()


#mở Stream


stream = p.open(format=FORMAT,
                
                channels = CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Đang thu âm vào buffter....")


try:
    while True:
        # 1. Đọc dữ liệu thô từ card âm thanh
        data = stream.read(CHUNK)
        
        # 2. Chuyển đổi sang mảng NumPy để tính toán (VAD, STT)
        audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # 3. Nạp vào buffer chính
        audio_buffer.append(audio_chunk)
        
        # Kiểm tra nếu buffer đủ dài (ví dụ 1 giây) thì đẩy đi xử lý
        if len(audio_buffer) > (RATE / CHUNK): 
            process_audio = np.concatenate(audio_buffer)
            # Gọi hàm STT của bạn ở đây: STT_chunk_whisper(process_audio)
            audio_buffer = [] # Reset buffer sau khi xử lý xong một đoạn
except KeyboardInterrupt:
    print("Dừng thu âm.")
    stream.stop_stream()
    stream.close()
    p.terminate()