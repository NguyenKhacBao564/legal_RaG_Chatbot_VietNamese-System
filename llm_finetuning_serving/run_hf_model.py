"""
Script to run the fine-tuned Vietnamese Legal LLM directly from Hugging Face
"""

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

def run_model():
    model_id = "NguyenBao564/vietnamese-legal-llama-3.1-8b"
    base_model_id = "unsloth/Llama-3.1-8B-Instruct"
    
    print(f"🚀 Loading base model: {base_model_id}")
    # Using 4-bit quantization for efficiency if bitsandbytes is installed
    try:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
    except ImportError:
        print("⚠️ bitsandbytes not found, loading in full precision (may require more VRAM)")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            device_map="auto",
            trust_remote_code=True
        )

    print(f"🚀 Loading PEFT adapter: {model_id}")
    model = PeftModel.from_pretrained(base_model, model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Set up streamer for real-time output
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    print("\n✅ Model loaded successfully! (Type 'quit' or 'exit' to stop)\n")

    # System prompt for Vietnamese legal expert
    system_prompt = (
        "Bạn là một chuyên gia tư vấn pháp luật Việt Nam với nhiều năm kinh nghiệm. "
        "Hãy trả lời các câu hỏi một cách chính xác, chi tiết và dễ hiểu. "
        "Luôn dẫn nguồn từ các văn bản pháp luật cụ thể khi có thể."
    )

    while True:
        user_input = input("❓ Câu hỏi pháp luật của bạn: ")
        if user_input.lower() in ["quit", "exit"]:
            break
        
        # Format for Llama-3.1 Instruct
        prompt = (
            f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{user_input}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        print("\n🤖 Trả lời: ", end="")
        _ = model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    run_model()
