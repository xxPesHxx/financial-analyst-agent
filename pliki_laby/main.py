import os, time
from platform import system
from typing import List

import torch, google.generativeai as genai
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)
BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-20b")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flesh")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")

tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_NAME,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
client = OpenAI(
    api_key= GROQ_API_KEY,
    base_url=BASE_URL,
)
def chat_once_openai(prompt: str, system: str, temperatura: float=0.3, top_p: float=0.95, top_k: int=40, max_output_tokens: int=512):
    t0 = time.time()
    response = client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=[{"role": "user", "content": prompt},
                  {"role": "system", "content": system}
                  ],
        temperature=temperatura,
        top_p=top_p,
        max_tokens=max_output_tokens
    )
    dt = time.time() - t0
    choice = response.choices[0]
    out = {
        "text": choice.message.content,
        'finish_reason': choice.finish_reason,
        'latency_s' : round(dt, 3),
        'usage': getattr(response, 'usage', None) and response.usage.model_dump()
    }
    return out

def approx_tokens_openai(texts: List[str], model_name: str="gpt-40-mini") -> int:
    try:
        enc = tiktoken.encoding_for_model(model_name)
        return sum(len(enc.encode(t)) for t in texts)
    except Exception as e:
        return sum(max(1, len(t) // 4) for t in texts)
def estimate_cost_use(prompt_tokens: int, completion_tokens: int, price_in: float = 0.000005, price_out: float = 0.0000015) -> float:

    return prompt_tokens * price_in + completion_tokens * price_out



input_txt = "jaki sposób rozbudować mieśnie brzucha najbardziej efektywnie"

result = chat_once_openai(
    "w jaki sposób rozbudować mieśnie brzucha najbardziej efektywnie",
    "jesteś pomocnym trenerem odpowiadaj, zwięźle po polsku",
    1,
    0.95,
    40,
    512
)

output_txt = result['text']
ptoks = approx_tokens_openai('input_txt')
otoks = approx_tokens_openai('output_txt')

print(ptoks, otoks)
print(estimate_cost_use(ptoks, otoks))
print(result['usage'])
"""
print((result['text']))
print("finish_reason", (result['finish_reason']))
print("latency_s", (result['latency_s']))
print("usage", (result['usage']))
"""





"""
def chat_once_gemini(prompt: str, system: str="You are helpful assistant.",
                    temperatura: float=0.3, top_p: float=0.95, top_k: int=40, max_output_tokens: int=512) -> str:
    config = genai.GenerationConfig(
        temperature=temperatura,
        top_k=top_k,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        stop_sequences=["<END>"]
    )
    t0 = time.time()

    model = genai.GenerativeModel(
        GEMINI_MODEL,
        system_instruction=system,
        generation_config=config

    )
    resp = model.generate_content(prompt)
    dt = time.time() - t0
    return {
        "text": getattr(resp, "text", str(resp)),
        "latency_s": dt,
        "usage": {
            "completion_token": resp.usage_metadata.candidates_token_count,
            "prompt_token_count": resp.usage_metadata.prompt_token_count,
            "total_token": resp.usage_metadata.total_token_count
        }
    }

result = chat_once_gemini(
    "w jaki sposób rozbudować mieśnie brzucha najbardziej efektywnie",
    "jesteś pomocnym trenerem odpowiadaj, zwięźle po polsku",
    1,
    0.95,
    40,
    512
)

print((result['text']))


"""