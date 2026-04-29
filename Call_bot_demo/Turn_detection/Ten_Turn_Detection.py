from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

local_path = "/home/linhhv/working/ten_turn_detection_local"
# model_id = 'TEN-framework/TEN_Turn_Detection'
# model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.bfloat16)
# tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
# model.save_pretrained(local_path)
# tokenizer.save_pretrained(local_path)
# Nạp Tokenizer từ local
tokenizer = AutoTokenizer.from_pretrained(
    local_path, 
    trust_remote_code=True,
)

# Nạp Model từ local
model = AutoModelForCausalLM.from_pretrained(
    local_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    local_files_only=True # Ép buộc chỉ dùng file ở local, không check internet
).to("cuda") # Đưa lên GPU

model.eval()


# Function for inference
def analyze_text(text, system_prompt=""):
    inf_messages = [{"role":"system", "content":system_prompt}] + [{"role":"user", "content":text}]
    input_ids = tokenizer.apply_chat_template(
        inf_messages, 
        add_generation_prompt=True, 
        return_tensors="pt"
    ).cuda()
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids, 
            max_new_tokens=1, 
            do_sample=True, 
            top_p=0.1, 
            temperature=0.1, 
            pad_token_id=tokenizer.eos_token_id
        )
        
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)


# text = "Xin chào, tôi có một câu hỏi về"
# result = analyze_text(text)

# print(f"Input: '{text}'")
# print(f"Turn Detection Result: '{result}'")

