import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")

print(f"Ładowanie modelu: {LOCAL_MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print("Urządzenie:", device)

def generate_local(prompt: str, system: str = "You are a helpful assistatnt.", max_new_tokens: int = 128,
                   temperature: float = 0.0, top_p: float = 0.9):
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        model_inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        ).to(device)
    else:
        text = f"[SYSTEM]\n{system}\n[USER]\n{prompt}\n[ASSISTANT]"
        model_inputs = tokenizer(
            text,
            return_tensors="pt",
        )
    with torch.no_grad():
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0.0),
            temperature=temperature if temperature > 0.0 else None,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id

        )
    if hasattr(tokenizer, "apply_chat_template"):
        gen_ids = output_ids[:, model_inputs["input_ids"].shape[-1]:]
    else:
        gen_ids = output_ids

    text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    return text

#try:
#   out = generate_local("Podaj 3 krotkie pomysły na aktywność fizyczną w domu.")
#   print(out)
#except Exception as e:
#    print("Bład generatora lokalnego:", e)

def count_tokens_local(text: str) -> int:
    return len(tokenizer.encode(text))
prompt = "Stwórz 3 punktową listę porad jak uczyć się efektywnie."
sysmsg = "You are a concise tutor."
ptoks = count_tokens_local(prompt) + count_tokens_local(sysmsg)
loc_out = generate_local(
    prompt=prompt,
    system=sysmsg,
    max_new_tokens=120,
    temperature=0.2,
    top_p=0.9,
)
"""
ctoks = count_tokens_local(loc_out)
print("Prompt tokens:", ptoks, "} Completion tokens:", ctoks)
print("--- Odpowiedź ---\n", loc_out)
"""
import numpy as np

def softmax(logits):
    logits = np.asarray(logits, dtype=float)
    m = np.max(logits)
    exps = np.exp(logits - m)
    probs = exps / np.sum(exps)
    return probs

def softmax_with_temperature(logits, temperature: float = 1.):
    logits = np.asarray(logits, dtype=float)
    T = float(temperature)
    if T <= 0:
        raise ValueError("temperature must be > 0")
    scaled = logits / T
    m = np.max(scaled)
    exps = np.exp(scaled - m)
    probs = exps / np.sum(exps)
    return probs

def top_k_mask(probs, k: int | None):
    probs = np.asarray(probs, dtype=float)
    if k is None or k >= probs.size:
        return np.ones_like(probs, dtype=bool)
    idx = np.argsort(-probs)[:k]
    mask = np.zeros_like(probs, dtype=bool)
    mask[idx] = True
    return mask

def top_p_mask(probs, top_p: float = 1.0):
    probs = np.asarray(probs, dtype=float)
    p = float(top_p)
    if p >= 1.0:
        return np.ones_like(probs, dtype=bool)
    if p <= 0.0:
        raise ValueError("top_p must be in (0, 1]")
    order = np.argsort(-probs)
    sorted_probs = probs[order]
    csum = np.cumsum(sorted_probs)
    cutoff_idx = np.searchsorted(csum, p, side="left")
    keep = order[: cutoff_idx + 1]
    mask = np.zeros_like(probs, dtype=bool)
    mask[keep] = True
    return mask

def print_array(arr):
    print(" ".join(f"{v:0.3f}" for v in arr))

def renorm(probs, mask):
    masked = probs * mask
    s = masked.sum()
    if s <= 0:
        m = np.argmax(probs)
        out = np.zeros_like(probs)
        out[m] = 1.0
        return out
    return masked / s



logits = [2.0, 1.0, 0.0, -0.5, -1.0]

print_array(logits)
probs = softmax(logits)
print_array(probs)
probs_tmp = softmax_with_temperature(logits, 0.4)
print_array(probs_tmp)
mask_k = top_k_mask(probs_tmp, 3)
mask_p = top_p_mask(probs_tmp, 0.7)
print(mask_k)
print(mask_p)
renorm_values = renorm(probs, mask_k)
print_array(renorm_values)

def sample_next_token(
    logits,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int | None = None,
    rng: np.random.Generator | None = None,
):
    if rng is None:
        rng = np.random.default_rng()
    simple_probs = softmax(logits)
    probs = softmax_with_temperature(logits, temperature=temperature)
    mask_k = top_k_mask(probs, top_k)
    mask_p = top_p_mask(probs, top_p)
    mask = mask_k & mask_p
    probs_f = renorm(probs, mask)
    idx = rng.choice(len(probs_f), p=probs_f)
    return idx, simple_probs, probs, probs_f, mask


logits = [2.0, 1.0, 0.0, -0.5, -1.0]  # przykładowe logity
idx, simple_probs, raw_probs, filtered_probs, mask = sample_next_token(
    logits,
    temperature=0.3,
    top_p=0.95,   # lub 1.0, jeśli bez nukleusa
    top_k=None,   # lub np. 40 dla lokalnych LLM
)
print("Logits:")
print_array(logits)
print("Simple probs (softmax):")
print_array(simple_probs)
print("Raw probs (softmax with temperature):")
print_array(raw_probs)
print("Filtered probs:")
print_array(filtered_probs)
print("Mask:", mask)
print("Sampled token index:", idx)