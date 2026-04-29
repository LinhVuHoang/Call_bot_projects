import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))

from VAD.Silero import get_vad_timestamps
import librosa
from Voice_gender_detection.voice_detection import classify_audio_chunk
from STT_VN.Whisper_medium import STT_chunk_whisper
from Normalization_data.xlm_roberta_capu import normalize_text
from Turn_detection.Ten_Turn_Detection import analyze_text
import string
import noisereduce as nr
from LLM.open_ai_llm import generate_response
from TTS.viXTTS_voice import generalize_tts_original_text
import asyncio

audio_path="/home/linhhv/working/lvh/Call_Bot/Call_bot_demo/STT_VN/record.wav"
segments = get_vad_timestamps(audio_path)
translator = str.maketrans('', '', string.punctuation)

system_prompt ="""

Bạn là một trợ lý tổng đài qua cuộc gọi thoại.

Nhiệm vụ của bạn là trả lời câu hỏi của khách hàng một cách lịch sự, rõ ràng và ngắn gọn như khi đang nói chuyện qua điện thoại.

Quy tắc:
Trả lời tự nhiên như đang nói chuyện, giọng điệu thân thiện và chuyên nghiệp.
Câu trả lời ngắn gọn, rõ ý, dễ hiểu.
Luôn giữ thái độ lịch sự.
Nếu câu hỏi chưa rõ, hãy hỏi lại nhẹ nhàng để làm rõ thông tin.
Không sử dụng markdown, ký hiệu đặc biệt hoặc danh sách.
Câu trả lời sẽ được đọc bằng hệ thống text to speech nên cần viết theo văn nói tự nhiên, tránh câu quá dài hoặc phức tạp.
Tất cả số, phép toán và ký hiệu phải được viết bằng chữ.
Các từ viết tắt tiếng anh phải được phiên âm sang tiếng việt, ví dụ ây ai.
Hạn chế dùng từ tiếng anh, ưu tiên tiếng việt tự nhiên.
Tránh trả lời quá dài, ưu tiên một đến ba câu.
"""

#Changing file path to real-time call




def cut_audio(audio,start,end,sr):
    start_sample = int(start*sr)
    end_sample = int(end*sr)
    return audio[start_sample:end_sample]





async def response_audio_streaming(full_audio_np, sample_rate=16000):
    """
    Hàm xử lý audio nhận trực tiếp từ buffer WebSocket
    full_audio_np: Mảng numpy float32 nhận từ np.concatenate(audio_buffer)
    """
    # 1. Khử nhiễu (Sử dụng asyncio.to_thread vì nr.reduce_noise tốn CPU)
    audio = await asyncio.to_thread(nr.reduce_noise, y=full_audio_np, sr=sample_rate) 
    
    audio_duration = len(audio) / sample_rate
    buffer_text = ""
    
    # 2. VỚI STREAMING: 
    # Nếu bạn đã gom đủ một câu (phát hiện bằng VAD ở ngoài), 
    # bạn có thể chạy STT trên toàn bộ audio này thay vì cắt segment nhỏ.
    
    # Chạy STT cho cả cụm âm thanh đã nhận được
    text_result = await asyncio.to_thread(STT_chunk_whisper, audio, sample_rate)
    # --- ĐOẠN SỬA LỖI TẠI ĐÂY ---
    if isinstance(text_result, (list, tuple)) and len(text_result) > 0:
        raw_text = str(text_result[0]).strip()
    elif isinstance(text_result, str):
        raw_text = text_result.strip()
    else:
        raw_text = ""
    # ----------------------------
    # # Lấy text thô (Whisper thường trả về list/tuple)
    # raw_text = text_result[0].strip() if text_result else ""
    # raw_text ="Tôi muốn hỏi về khóa học"
    
    if not raw_text:
        print("⚠️ Không nhận diện được văn bản.")
        return

    buffer_text = raw_text
    print(f"📝 Nội dung nhận diện: {buffer_text}")

    # 3. Phân loại giới tính (Dùng sample đã khử nhiễu)
    result = await asyncio.to_thread(classify_audio_chunk, audio, sample_rate)
    max_key = max(result, key=result.get)
    gender = "Nam" if max_key == 'male' else "Nữ"
    print(f"👤 Giới tính: {gender} ({result[max_key]:.2f})")

    # 4. Kiểm tra Turn Detection (Kết thúc câu hay chưa)
    # Vì logic VAD ở WebSocket đã xác định "Ngừng nói", 
    # ở đây ta coi như turn đã "finished" hoặc kiểm tra thêm bằng AI.
    turn = await asyncio.to_thread(analyze_text, buffer_text)
    
    # 5. Xử lý phản hồi AI
    # Lưu ý: 'input_ms' nên là buffer_text thay vì câu cứng
    # system_prompt = "Bạn là trợ lý ảo hỗ trợ khóa học..." # Đảm bảo biến này đã có
    
    print("🎯 Đang tạo phản hồi LLM...")
    response_user = await asyncio.to_thread(
        generate_response, system_prompt, buffer_text, gender
    )
    
    print(f"🤖 Bot: {response_user}")

    # # 6. Chuyển sang TTS (Phát âm thanh)
    # await asyncio.to_thread(generalize_tts_original_text, response_user)
    
    print("✅ Xử lý hoàn tất!")
    return response_user, audio_duration, raw_text
        

                
                
# asyncio.run(response_audio(audio_path))
    
    


    