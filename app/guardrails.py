import re

class SecurityError(Exception):
    pass

def detect_injection(text: str) -> bool:
    INJECTION_PATTERNS = [
        r'ignore (all|previous|above) instructions',
        r'reveal (system|developer) prompt',
        r'jailbreak',
        r'override .* rules',
        r'system prompt',
        r'zignoruj instrukcje',
        r'ignoruj instrukcje',
        r'zasady'
    ]
    return any(re.search(p, text.lower()) for p in INJECTION_PATTERNS)

def validate_ticker(ticker: str) -> bool:
    match = re.match(r'^[A-Z]{3,5}$', ticker)
    if not match:
        raise SecurityError(f"Invalid ticker format: {ticker}")
    return True

def validate_output(text: str) -> bool:
    RESTRICTED_KEYWORDS = ["CONFIDENTIAL", "SECRET_KEY", "password"]
    for keyword in RESTRICTED_KEYWORDS:
        if keyword in text:
            raise SecurityError(f"Output contains restricted data: {keyword}")
    return True

def sanitize_path(path: str) -> str:
    if ".." in path or path.startswith("/") or "\\" in path:
        raise SecurityError("Path traversal detected")
    return path
