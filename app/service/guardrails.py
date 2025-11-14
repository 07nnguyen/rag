from fastapi import HTTPException
_BLOCKLIST = ["ignore previous", "system prompt", "bypass", "delete instructions",]
def guard_input(text: str):
    low = text.lower()
    if any(k in low for k in _BLOCKLIST):
        raise HTTPException(400, detail="Input rejected by guardrails.")

def guard_output(text: str) -> str:
# Ensure some minimal constraints; extend with PII filters, JSON schema,
    if len(text) > 20000:
        return text[:20000] + "\n\n[truncated]"
    return text