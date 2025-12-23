import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

from typing import Union
Prompt = Union[str, list[int]]

class InferenceEngine:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        self.model.eval()
        self.device = device

    def generate(self, prompts: list[Prompt], sampling_params):
        # max_new_tokens: int = 256, temperature: float = 0.6, top_p: float = 0.9
        results = []

        for prompt in prompts:
            if isinstance(prompt, str):
                enc = self.tokenizer(prompt, return_tensors="pt")
                input_ids = enc["input_ids"].to(self.device)
                attention_mask = enc["attention_mask"].to(self.device)
            else:
                input_ids = torch.tensor([prompt], dtype=torch.long, device=self.device)
                attention_mask = torch.ones_like(input_ids, device=self.device)
        
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=sampling_params.max_new_tokens,
                do_sample=(sampling_params.temperature > 0),
                temperature=sampling_params.temperature,
                top_p=sampling_params.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            new_tokens = out[0, input_ids.shape[1]:].tolist()
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            results.append({
                "text": text,
                "token_ids": new_tokens,
            })
        return results
