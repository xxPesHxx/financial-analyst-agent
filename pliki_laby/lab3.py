import glob, re, time, json, os, torch

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Any, Dict, Optional, Tuple, Literal
from dotenv import load_dotenv


class CalcArg(BaseModel):
    a: float
    b: float


class ConvertArgs(BaseModel):
    value: float
    from_unit: Literal["km", "mi", "c", "f"]
    to_unit: Literal["km", "mi", "c", "f"]


class SearchArgs(BaseModel):
    pattern: str = Field(..., max_length=64)


class KBArgs(BaseModel):
    query: str
    top_k: int = Field(3, ge=1, le=10)


class ToolCall(BaseModel):
    tool: Literal["calculator.add", "calculator.sub", "calculator.mul", "calculator.div",
    "units.convert", "files.search", "kb.lookup"]
    args: Dict[str, Any]


ALLOWED_TOOLS = {
    "calculator.add",
    "calculator.sub",
    "calculator.mul",
    "calculator.div",
    "units.convert",
    "files.search",
    "kb.lookup"
}


def _run_tool_sync(tc: ToolCall) -> Dict[str, Any]:
    if tc.tool not in ALLOWED_TOOLS:
        raise ValueError("Tool not allowed: " + tc.tool)
    if tc.tool.startswith("calculator."):
        args = CalcArg(**tc.args)
        op = tc.tool.split(".")[1]
        if op == "add":
            res = args.a + args.b
        elif op == "sub":
            res = args.a - args.b
        elif op == "mul":
            res = args.a * args.b
        elif op == "div":
            if args.b == 0:
                raise ValueError("Division by zero")
            res = args.a / args.b
        else:
            raise ValueError("Unknown calculator operation: " + op)
        return {"result": res}
    if tc.tool == "units.convert":
        args = ConvertArgs(**tc.args)
        if args.from_unit == args.to_unit:
            return {"result": args.value}
        if args.from_unit == "km" and args.to_unit == "mi":
            return {"result": args.value * 0.621371}
        if args.from_unit == "mi" and args.to_unit == "km":
            return {"result": args.value / 0.621371}
        if args.from_unit == "c" and args.to_unit == "f":
            return {"result": args.value * 9 / 5 + 32}
        if args.from_unit == "f" and args.to_unit == "c":
            return {"result": (args.value - 32) * 5 / 9}
        raise ValueError("Unsupported unit conversion")
    if tc.tool == "files.search":
        args = SearchArgs(**tc.args)
        pattern = args.pattern
        matched_files = []
        for filepath in glob.glob("./**/*", recursive=True):
            if os.path.isfile(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if re.search(pattern, content):
                            matched_files.append(filepath)
                except Exception:
                    continue
        return {"files": matched_files[:10]}
    if tc.tool == "kb.lookup":
        args = KBArgs(**tc.args)
        hits = kb_lookup(args.query, top_k=args.top_k)
        return {"hits": hits}
    raise ValueError("Unhandled tool")


def run_tool(tc: ToolCall, timeout_s: float = 2.0):
    with ThreadPoolExecutor(max_workr=1) as ex:
        fut = ex.submit(_run_tool_sync, tc)
        try:
            out = fut.result(timeout=timeout_s)
            return True, out, None
        except FuturesTimeout:
            return False, {}, "timeout"
        except Exception as e:
            return False, {}, str(e)


KB_PATH = './lab03_kb.json'
print("KB path:", KB_PATH, "exists:", os.path.exists(KB_PATH))


def load_kb(path: str = KB_PATH) -> List[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


KB = load_kb() if os.path.exists(KB_PATH) else []
print("Loaded KB entries:", len(KB))

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
USE_LOCAL = os.getenv("LLM_LOCAL", "0") == "1"
tokenizer = model = None

load_dotenv()

if USE_LOCAL:
    LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_NAME,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Using local model: {LOCAL_MODEL_NAME} on device: {device}")
else:
    from google import genai

    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    print(f"Using Gemini model: {GEMINI_MODEL} via Google GenAI")


def chat_once_gemini(
        prompt: str,
        system: str = "You are a helpful assistant.",
        temperature: float = 0.3,
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


def count_tokens_local(texts: List[str]) -> List[int]:
    if tokenizer is None:
        raise ValueError("Tokenizer is not initialized for local model.")
    return [len(tokenizer.encode(text)) for text in texts]


def generate_local(prompt: str, system: str = "You are a helpful assistant.", max_new_tokens: int = 128,
                   temperature: float = 0.0, top_p: float = 0.9) -> str:
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
        text = f"[SYSTEM]\n{system}\n[USER]\n{prompt}\n[ASSISTANT]\n"
        model_inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0.0),
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if temperature > 0.0:
            gen_kwargs["temperature"] = temperature

        output_ids = model.generate(
            **model_inputs,
            **gen_kwargs
        )

    if hasattr(tokenizer, "apply_chat_template"):
        gen_ids = output_ids[:, model_inputs['input_ids'].shape[-1]:]
    else:
        gen_ids = output_ids

    text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    return text


def chat_once_local(
        prompt: str,
        system: str = "You are a helpful assistant.",
        temperature: float = 0.3,
        top_p: float = 0.95,
        max_output_tokens: int = 1024
) -> Dict[str, Any]:
    t0 = time.time()
    text = generate_local(
        prompt,
        system=system,
        max_new_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p
    )
    dt = round(time.time() - t0, 3)
    prompt_tokens_count = count_tokens_local([prompt])[0]
    completion_tokens_count = count_tokens_local([text])[0]
    return {
        "text": text,
        "latency_s": dt,
        "usage": {
            "prompt_tokens": prompt_tokens_count,
            "completion_tokens": completion_tokens_count,
            "total_tokens": prompt_tokens_count + completion_tokens_count
        }
    }


def llm_call(
        prompt: str,
        system: str = "You are a helpful assistant.",
        temperature: float = 0.3,
        top_p: float = 0.95,
        top_k: int = 40,
        max_output_tokens: int = 1024,
) -> Dict[str, Any]:
    if USE_LOCAL:
        return chat_once_local(
            prompt,
            system=system,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens
        )
    else:
        return chat_once_gemini(
            prompt,
            system=system,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_output_tokens
        )


def normalize(text: str) -> List[str]:
    text = text.lower()
    tokens = re.findall(r"[a-ząćęłńóśźż0-9]+", text, flags=re.IGNORECASE)
    return tokens


def score_entry(query_tokens: List[str], entry: Dict[str, Any]) -> float:
    c_tokens = normalize(entry.get("content", ""))
    t_tokens = normalize(entry.get("title", ""))
    tags = [normalize(t) for t in entry.get("tags", [])]
    tags_flat = [t for sub in tags for t in sub]
    base = sum(1 for q in query_tokens if q in c_tokens) * 1.0
    title_bonus = sum(1 for q in query_tokens if q in t_tokens) * 1.5
    tags_flat = sum(1 for q in query_tokens if q in tags_flat) * 1.2
    return base + title_bonus + tags_flat


def kb_lookup(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    if not KB:
        return []
    query_tokens = normalize(query)
    scored_entries: List[Tuple[float, Dict[str, Any]]] = []
    for entry in KB:
        score = score_entry(query_tokens, entry)
        if score > 0:
            scored_entries.append((score, entry))
    scored_entries.sort(key=lambda x: x[0], reverse=True)
    top_entries = [entry for score, entry in scored_entries[:max(1, top_k)]]
    return top_entries


print(kb_lookup("What is token?", top_k=2))

