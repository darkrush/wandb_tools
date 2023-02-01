import json
from typing import Dict, Any

with open('config.json') as f:
    CONFIG_DICT: Dict[str, Any] = json.load(f)

assert 'CARED_KEY_LIST' in CONFIG_DICT.keys()
