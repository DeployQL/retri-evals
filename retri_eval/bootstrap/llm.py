from llama_cpp import Llama
from dsp.modules.lm import LM
import requests

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


class LLMServer(LM):
    def __init__(self, name="server", url="http://192.168.0.17:8888/completion"):
        super().__init__(model=name)
        self.url = url
        self.model = name

    def basic_request(self, prompt: str, **kwargs):
        # resp = requests.post(self.url, json={
        #     "prompt": prompt,
        #     "n_predict": kwargs.get("n_predict", 128)
        # })
        # return resp.json()['content']
        resp = requests.post(
            self.url,
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "model": self.model,
                "n_predict": kwargs.get("n_predict", 256),
            },
        )
        return [resp.json()["choices"][0]["message"]["content"]]

    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        return self.basic_request(prompt, **kwargs)
