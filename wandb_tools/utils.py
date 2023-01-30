from typing import List, Dict, Any
import copy


def dict_equal(configa: Dict, configb: Dict, exclude_list: List[str]):
    configa_copy = copy.deepcopy(configa)
    configb_copy = copy.deepcopy(configb)
    for exclude_key in exclude_list:
        configa_copy.pop(exclude_key, None)
        configb_copy.pop(exclude_key, None)
    return configa_copy == configb_copy


def except_seed(configa: Dict, configb: Dict):
    return dict_equal(configa, configb, ['seed'])


def diff_config_list_old(config_list: List[Dict],
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


def diff_configs(config_list: List[Dict],
                 key_list: List[str]):
    diff_key_list = []
    for main_key in key_list:
        different_value_list = []
        for config in config_list:
            config_value = config[main_key] if main_key in config.keys() else None
            if config_value not in different_value_list:
                different_value_list.append(config_value)
        assert len(different_value_list) > 0
        if len(different_value_list) > 1:
            diff_key_list.append(main_key)
    return diff_key_list


def diff_config(config_A: Dict[str, Any], config_B: Dict[str, Any],
                key_list: List[str]):
    diff_key_list = []
    for key in key_list:
        v_a = config_A.get(key, None)
        v_b = config_B.get(key, None)
        if not v_a == v_b:
            diff_key_list.append(key)
    return diff_key_list
