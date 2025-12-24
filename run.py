import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from ohmyvllm.engine.inference_engine import InferenceEngine
from ohmyvllm import SamplingParams

def main():
    path = "Qwen/Qwen3-0.6B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = InferenceEngine(path, device=device)

    sampling_params = SamplingParams(max_new_tokens=256,temperature=0.6)
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
    outputs = model.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")

if __name__ == "__main__":
    main()