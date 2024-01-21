from llama_cpp import Llama
from dsp.modules.lm import LM

# from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import torch

# DefaultQueryLLM = Llama(model_path="/home/mtbarta/deployql/reps/results/llama-2-7b.Q4_K_M.gguf", n_gpu_layers=-1, max_tokens=150, echo=False)
# TheBloke/Mixtral-8x7B-v0.1-GGUF
# DefaultQueryLLM = Llama(model_path="/home/mtbarta/deployql/reps/results/llama-2-7b.Q4_K_M.gguf", n_gpu_layers=-1, max_tokens=150, echo=False)


class DefaultLM(LM):
    def __init__(self):
        super().__init__(model="llama-2-7b")

        model_name_or_path = "TheBloke/Llama-2-7B-fp16"

        # Load model
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        self.llm = pipe = pipeline(
            "text-generation",
            device=-1,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            repetition_penalty=1.1,
        )

    def basic_request(self, prompt: str, **kwargs):
        return self.llm(prompt)[0]["generated_text"]

    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        return self.llm(prompt)[0]["generated_text"]
