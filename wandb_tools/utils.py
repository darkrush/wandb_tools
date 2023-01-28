from typing import List, Dict
import numpy as np


def except_seed(config1: Dict, config2: Dict):
    config1_copy = config1.copy()
    if 'seed' in config2.keys():
        config1_copy['seed'] = config2['seed']
    else:
        return False
    return config1_copy == config2


def diff_config_list(config_list: List[Dict],
                     exclude_list: List[str] = ['seed']):
    all_key_list = []
    diff_key_list = []
    for config in config_list:
        for key in config.keys():
            if (key not in all_key_list) and (key not in exclude_list):
                all_key_list.append(key)
    for main_key in all_key_list:
        different_value_list = []
        for config in config_list:
            config_value = config[main_key] if main_key in config.keys() else None
            if config_value not in different_value_list:
                different_value_list.append(config_value)
        assert len(different_value_list) > 0
        if len(different_value_list) > 1:
            diff_key_list.append(main_key)
    return diff_key_list
