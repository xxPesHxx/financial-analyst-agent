import os, time
from http.client import responses

from openai import OpenAI
from typing import Dict, Any, Optional, TypedDict
from google import genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))

def chat_once_gemini(
    prompt: str,
    system: str = "You are a helpful assistant.",
    temperature: float = 0.0,
    top_p: float = 0.95,
    top_k: int = 40,
    max_output_tokens: int = 1024,
) -> Dict[str, Any]:
    config = genai.types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_output_tokens=max_output_tokens,
        stop_sequences=["<END>"],
    )

    t0 = time.time()
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=config,
    )
    dt = round(time.time() - t0, 3)

    usage = getattr(resp, "usage_metadata", None)
    usage_dict: Optional[Dict[str, Any]] = None
    if usage is not None:
        usage_dict = {
            "prompt_tokens": getattr(usage, "prompt_token_count", None),
            "completion_tokens": getattr(usage, "candidates_token_count", None),
            "total_tokens": getattr(usage, "total_token_count", None),
            "tool_use_prompt_tokens": getattr(usage, "tool_use_prompt_token_count", None),
            "thoughts_tokens": getattr(usage, "thoughts_token_count", None),
        }

    return {
        "text": getattr(resp, "text", str(resp)),
        "latency_s": dt,
        "usage": usage_dict
    }

prompts = [
    "Summarize this text.",
    "Summarize this text in one sentence.",
    "Summarize this text in one sentence using simple ENglish and output as JSON {summary: ...}"
]

text = """
    Artificial intelligence (AI) is a field of computer science that builds systems able to perform tasks that typically require human intelligence-such as understanding language, learning from data, reasoning, and perception.
    What AI can do: perception (vision/speech), reasoning, learning, interaction (natural language), and planning/control.
    How it works (at a glance):
    - Symbolic AI: hand-written rules and logic.
    - Machine learning: models learn patterns from data.
    - Deep learning: multi-layer neural networks for images, speech, and text.
"""

"""
for p in prompts:
    print("\n---\nPrompt: ", p)
    responses = chat_once_gemini(f"{p}\n\n{text}")
    print("---\nAnswer: ", responses["text"])
    print(f"---\n {responses['latency_s']} s | Tokens: {responses['usage']['total_tokens']}")
"""
"""
roles = [
    "You are a sarcastic assistant.",
    "you are a formal university lecturer.",
    "You are a motivational coach."
]

question = "Explain recursion in one sentence."

for r in roles:
    print("\n---\nRole: ", r)
    print("---\nQuestion: ", question)
    responses = chat_once_gemini(f"{question}", system=r, temperature=0.3)
    print("---\nAnswer: ", responses["text"])
    print(f"---\n{responses['latency_s']} s | Tokens: {responses['usage']['total_tokens']}")
"""

question = """
    Translate English -> Polish:
    Input: Good morning -> Output: Dzień dobry
    Input: Thank you -> Output: Dziękuję
    Input: See you later -> Output:
"""


"""
print("---\nQuestion: ", question)
responses = chat_once_gemini(f"{question}", temperature=0.3)
print("---\nAnswer: ", responses["text"])
print(f"---\n {responses['latency_s']} s | Tokens: {responses['usage']['total_tokens']}")
"""

"""
question = "If there are 3 red and 5 blue balls, and you take one randomly, what is the probability it's red?"

print("---\nQuestion: ", question)
responses = chat_once_gemini(f"{question}", temperature=0.3)
print("---\nAnswer (without CoT): ", responses["text"])
print(f"---\n {responses['latency_s']} s | Tokens: {responses['usage']['total_tokens']}")

print("---\nQuestion: ", question)
responses = chat_once_gemini(f"{question}\n Think step by step.", temperature=0.3)
print("---\nAnswer: ", responses["text"])
print(f"---\n {responses['latency_s']} s | Tokens: {responses['usage']['total_tokens']}")
"""

import json
prompt = """
Classify the sentiment of the text as positive, negative, or neutral. Return JSON {sentiment: ...}
"""
text = "I love how easy this app is to use."
"""
try:
    responses = chat_once_gemini(f"{prompt}\n\n{text}", temperature=0.3)
    print("---\nAnswer: ", responses["text"])
    data = json.loads(responses["text"])
    print("Parsowanie JSON OK: ", data)
    print(f"---\n {responses['latency_s']} s | Tokens: {responses['usage']['total_tokens']}")
except json.JSONDecodeError as e:
    print("Błąd parsowania JSON.")
"""

from typing_extensions import TypedDict, Literal

class Sentiment(TypedDict):
    sentiment: Literal["positive", "negative", "neutral"]

def chat_once_gemini_v2(
    prompt: str,
    system: str = "You are a helpful assistant.",
    temperature: float = 0.0,
    top_p: float = 0.95,
    top_k: int = 40,
    max_output_tokens: int = 1024,
) -> Dict[str, Any]:
    config = genai.types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_output_tokens=max_output_tokens,
        stop_sequences=["<END>"],
        response_mime_type="application/json",
        # response_schema=Sentiment
        response_json_schema={
            "type": "object",
            "properties": {
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral"]
                }
            },
            "required": ["sentiment"],
        }
    )

    t0 = time.time()
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=config,
    )
    dt = round(time.time() - t0, 3)

    usage = getattr(resp, "usage_metadata", None)
    usage_dict: Optional[Dict[str, Any]] = None
    if usage is not None:
        usage_dict = {
            "prompt_tokens": getattr(usage, "prompt_token_count", None),
            "completion_tokens": getattr(usage, "candidates_token_count", None),
            "total_tokens": getattr(usage, "total_token_count", None),
            "tool_use_prompt_tokens": getattr(usage, "tool_use_prompt_token_count", None),
            "thoughts_tokens": getattr(usage, "thoughts_token_count", None),
        }

    return {
        "text": getattr(resp, "text", str(resp)),
        "latency_s": dt,
        "usage": usage_dict
    }

try:
    responses = chat_once_gemini_v2(f"{prompt}\n\n{text}", temperature=0.3)
    print("---\nAnswer: ", responses["text"])
    data = json.loads(responses["text"])
    print("Parsowanie JSON OK: ", data)
    print(f"---\n {responses['latency_s']} s | Tokens: {responses['usage']['total_tokens']}")
except json.JSONDecodeError as e:
    print("Błąd parsowania JSON.")
