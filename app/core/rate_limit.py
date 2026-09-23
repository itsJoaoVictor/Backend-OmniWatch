from slowapi import Limiter
from slowapi.util import get_remote_address
import os

redis_url = os.getenv('REDIS_URL')
if redis_url:
    limiter = Limiter(key_func=get_remote_address, storage_uri=redis_url)
else:
    limiter = Limiter(key_func=get_remote_address)
