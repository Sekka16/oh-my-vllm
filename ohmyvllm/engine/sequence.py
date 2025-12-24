from ohmyvllm.sampling_params import SamplingParams

class Sequence:
    def __init__(self, input_ids, sampling_params):
        self.input_ids = input_ids
        self.sampling_params = sampling_params
        self.generated_ids = []
        self.past = None
        self.is_finished = False

    @property
    def last_token(self):
        if self.generated_ids:
            return self.generated_ids[-1]
        return self.input_ids[-1]
    
    @property
    def all_token_ids(self):
        return self.input_ids + self.generated_ids
    
    def append(self, token_id, eos_token_id):
        token_id = int(token_id)
        self.generated_ids.append(token_id)
        if (not self.sampling_params.ignore_eos) and (token_id == eos_token_id):
            self.is_finished = True
        if len(self.generated_ids) >= self.sampling_params.max_new_tokens:
            self.is_finished = True

    def extend(self, token_ids, eos_token_id):
        for tid in token_ids:
            if self.is_finished:
                break
            self.append(tid, eos_token_id)