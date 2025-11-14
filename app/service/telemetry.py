import os
from langfuse import Langfuse

_langfuse = None

def get_lf() -> Langfuse | None:
    global _langfuse
    if _langfuse is not None:
        return _langfuse
    host = os.getenv("LANGFUSE_HOST")
    pk = os.getenv("LANGFUSE_PUBLIC_KEY")
    sk = os.getenv("LANGFUSE_SECRET_KEY")
    if not (host and pk and sk):
        return None
    _langfuse = Langfuse(host=host, public_key=pk, secret_key=sk,release=os.getenv("LANGFUSE_RELEASE"))
    return _langfuse
