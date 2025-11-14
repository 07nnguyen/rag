import os, json, hashlib
from typing import Optional
import redis

_REDIS = None
_TTL = int(os.getenv("CACHE_TTL_SECONDS", "600"))
def _get() -> redis.Redis:
    global _REDIS
    if _REDIS is None:
        _REDIS = redis.from_url(os.getenv("REDIS_URL", "redis://redis:6379/0"))
    return _REDIS

def _key(query: str, top_k: int, filters: Optional[dict]) -> str:
    raw = json.dumps({"q": query, "k": top_k, "f": filters}, sort_keys=True)
    return "rag:ans:" + hashlib.sha1(raw.encode()).hexdigest()

def get_cached_answer(query: str, top_k: int, filters: Optional[dict]) -> Optional[dict]:
    val = _get().get(_key(query, top_k, filters)) 
    return json.loads(val) if val else None

def set_cached_answer(query: str, top_k: int, filters: Optional[dict], data:dict):
    _get().setex(_key(query, top_k, filters), _TTL, json.dumps(data))   
