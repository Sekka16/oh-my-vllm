import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    path = "Qwen/Qwen3-0.6B"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path).to(device)
    model.eval()

    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]

    for p in prompts:
        enc = tokenizer(p, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=256,
            do_sample=True,      # 类似 temperature=0.6 的“随机采样”效果
            temperature=0.6,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

        new_tokens = out[0][input_ids.shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        print("\n")
        print(f"Prompt: {p!r}")
        print(f"Completion: {text!r}")

if __name__ == "__main__":
    main()