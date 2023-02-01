from typing import List, Dict, Any
import pickle
import wandb
import json
import click
import os
import copy
import logging
import shutil
import hashlib
import itertools
import pandas
import numpy as np
from wandb import sdk as wandb_sdk

from wandb.apis.public import File as wandb_File
from wandb_tools.utils import dict_equal, diff_config

logger = logging.getLogger(__name__)
api = wandb.Api()

CONFIG_NAME = 'config.json'
HISTORY_NAME = 'history.pkl'
META_NAME = 'wandb-metadata.json'
CARED_KEY_LIST = ['NT', 'NC', 'dataset', 'state_name',
                  'base_module', 'input_noise', 'control_name', 'lr']


class Siblings:
    def __init__(self, config: Dict[str, Any], command: str, 
                 exclude_keys: List[str] = ['seed']):
        self._config: Dict[str, Any] = copy.deepcopy(config)
        self._command: str = command
        self._exclude_keys: List[str] = copy.deepcopy(exclude_keys)
        for key in self._exclude_keys:
            self._config.pop(key, None)
        self._runID_list: List[str] = []

    @property
    def config_hash(self):
        hasher = hashlib.md5()
        config_json = json.dumps(self._config, sort_keys=True).encode('utf-8')
        hasher.update(config_json)
        return hasher.hexdigest()

    def check_in_siblings(self, config_b: Dict[str, Any]):
        return dict_equal(self._config, config_b, self._exclude_keys)

    def append_runID(self, runID: str):
        self._runID_list.append(runID)

    def __str__(self) -> str:
        return "config: {}\nrunID: {}\ncommand: {}".format(self._config, '|'.join(self._runID_list), self._command)

    def __repr__(self) -> str:
        return self.__str__()


class Group:
    def __init__(self, sib_list: List[Siblings], diff_key: str):
        self._sib_list = sib_list
        self._diff_key = diff_key

    def list_sibs_hash(self):
        return sorted([sib.config_hash for sib in self._sib_list])

    def __str__(self) -> str:
        return "key:{} sibID:{}".format(self._diff_key,
                                        [sib.config_hash for sib in self._sib_list])

    def __repr__(self) -> str:
        return self.__str__()


class Wandb_Local:
    def __init__(self, enterpoint: str, db_path: str) -> None:
        self._enterpoint = enterpoint
        self._db_path = db_path
        self._cache_path = os.path.join(self._db_path, self._enterpoint)
        self._local_runID_list = self._list_local_runID()
        self._siblings_dict: Dict[str, Siblings] = {}

    def _path_of_run(self, runID: str):
        return os.path.join(self._cache_path, runID)

    def _path_of_config(self, runID: str):
        return os.path.join(self._path_of_run(runID), CONFIG_NAME)

    def _path_of_meta(self, runID: str):
        return os.path.join(self._path_of_run(runID), META_NAME)

    def _path_of_history(self, runID: str):
        return os.path.join(self._path_of_run(runID), HISTORY_NAME)

    def _list_local_runID(self):
        local_runID_list: List[str] = []
        for item in os.scandir(self._cache_path):
            if item.is_dir():
                local_runID_list.append(item.name)
        local_runID_list.sort()
        return local_runID_list

    def _clean_cache(self):
        shutil.rmtree(self._cache_path)
        os.makedirs(self._cache_path)

    def list_runID(self):
        return self._local_runID_list

    def get_config(self, runID: str):
        config_path = self._path_of_config(runID)
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config

    def get_command(self, runID: str, exlude_seed: bool = True):
        meta_path = self._path_of_meta(runID)
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        args_list:List[str] = meta['args']
        if exlude_seed:
            seed_idx = args_list.index('--seed')
            args_list[seed_idx+1] = '${SEED}'
        args_str = ' '.join(args_list)
        codepath = meta['codePath']

        return '{} {}'.format(codepath, args_str)

    def get_history(self, runID: str):
        his_path = self._path_of_history(runID)
        with open(his_path, 'rb') as f:
            df_history: pandas.DataFrame = pickle.load(f)
        return df_history

    def clean_cache(self):
        confirm_str = 'Do you want to clean cache at path:{}?'.format(self._cache_path)
        confirm_clean = click.confirm(confirm_str, default=False)
        if confirm_clean:
            self._clean_cache()
        return confirm_clean

    def pull_runs(self, cache_clean: bool = False):
        runs = api.runs(self._enterpoint, per_page=100000)
        runs: List[wandb_sdk.wandb_run.Run]

        if not os.path.exists(self._cache_path):
            os.makedirs(self._cache_path)

        if cache_clean:
            self.clean_cache()
        remote_run_list = []
        for run in runs:
            runID = run.id
            remote_run_list.append(runID)
            if run.job_type in ['Alert']:
                continue
            if run.state not in ['finished']:
                continue
            logger.debug("check for {}".format(runID))
            local_run_dir = self._path_of_run(runID)
            config_path = self._path_of_config(runID)
            meta_path = self._path_of_meta(runID)
            feather_path = self._path_of_history(runID)

            need_update_flag_dict = {}
            item_dict = {'config': config_path,
                         'history': feather_path,
                         'meta': meta_path}
            if runID in self._local_runID_list:
                for key, path in item_dict.items():
                    need_update_flag_dict[key] = not os.path.exists(path)
            else:
                for key, path in item_dict.items():
                    need_update_flag_dict[key] = True

            if any(need_update_flag_dict.values()):
                update_list = [key for key, v in need_update_flag_dict.items() if v]
                logger.info("{} need to update {}".format(runID, update_list))

                if not os.path.exists(local_run_dir):
                    os.makedirs(local_run_dir)

                if need_update_flag_dict['config']:
                    config = run.config
                    with open(config_path, 'w') as f:
                        json.dump(config, f, indent=2)

                if need_update_flag_dict['history']:
                    config = run.config
                    df_history = run.history()
                    with open(feather_path, 'wb') as f:
                        pickle.dump(df_history, f)

                if need_update_flag_dict['meta']:
                    wandb_metafile: wandb_File = run.file("wandb-metadata.json")
                    temp_dir = os.path.join('/tmp', self._enterpoint)
                    file_IO = wandb_metafile.download(root=temp_dir, replace=True)
                    meta = json.load(file_IO)
                    with open(meta_path, 'w') as f:
                        json.dump(meta, f, indent=2)
                logger.info("{} updated".format(runID))
            else:
                logger.info("{} already cached in local".format(runID))

        for runID in self._local_runID_list:
            local_run_dir = self._path_of_run(runID)
            if runID not in remote_run_list:
                logger.info('{} not in remote but found in local,\
                             remove it'.format(runID))
                shutil.rmtree(local_run_dir)


def build_siblings(wandb_local: Wandb_Local):
    siblings_list: List[Siblings] = []
    for runID in wandb_local.list_runID():
        config = wandb_local.get_config(runID)
        command = wandb_local.get_command(runID)
        target_siblings = None
        for siblings in siblings_list:
            if siblings.check_in_siblings(config):
                target_siblings = siblings
                break
        if target_siblings is None:
            target_siblings = Siblings(config, command)
            siblings_list.append(target_siblings)
        target_siblings.append_runID(runID)
    return {sib.config_hash: sib for sib in siblings_list}


def find_all_groups(siblings_dict: Dict[str, Siblings]):
    sib_num = len(siblings_dict)
    hash_list = sorted(siblings_dict.keys())
    key_num = len(CARED_KEY_LIST)
    if sib_num == 0:
        raise ValueError('Siblings number is Zero.')
    con_mat = np.zeros([key_num, sib_num, sib_num], dtype=int)
    for key_idx in range(key_num):
        for idx1, idx2 in itertools.combinations(range(sib_num), 2):
            sib_1 = siblings_dict[hash_list[idx1]]
            sib_2 = siblings_dict[hash_list[idx2]]
            diff_key_list = diff_config(sib_1._config, sib_2._config,
                                        [CARED_KEY_LIST[key_idx]])
            con_mat[key_idx, idx1, idx2] = len(diff_key_list)
            con_mat[key_idx, idx2, idx1] = len(diff_key_list)
    group_list_dict: Dict[str, List[Group]] = {}
    unconvered_list = set(hash_list)
    for key_idx in range(key_num):
        idx_group_list = []
        for sib_idx in range(sib_num):
            target_group: List = None
            for group in idx_group_list:
                diff_num = np.sum(con_mat[:, sib_idx, group[0]])
                key_diff_num = con_mat[key_idx, sib_idx, group[0]]
                if not diff_num == key_diff_num:
                    continue
                else:
                    if key_diff_num == 0:
                        logger.warning('Two siblings different key number is zero')
                    target_group = group
                    break
            if target_group is None:
                idx_group_list.append([sib_idx])
            else:
                target_group.append(sib_idx)

        group_list: List[Group] = []
        for idx_group in idx_group_list:
            if len(idx_group) > 1:
                new_group = Group([siblings_dict[hash_list[sib_idx]]
                                   for sib_idx in idx_group],
                                  CARED_KEY_LIST[key_idx])
                group_list.append(new_group)
            else:
                continue
        group_list_dict[CARED_KEY_LIST[key_idx]] = group_list

        for group in group_list:
            unconvered_list.difference_update(group.list_sibs_hash())
    return group_list_dict, list(unconvered_list)



'''
    def get_metric_result(self,):
        result_dict: Dict[str, List[float]] = {}
        for runID in self._runID_list:
            his_path = self._parent_local._path_of_history(runID)
            with open(his_path, 'rb') as f:
                df_history: pandas.DataFrame = pickle.load(f)
            metric_key_list = [key for key in df_history.keys()
                               if key.startswith('loop_test')]
            for mk in metric_key_list:
                if mk in result_dict.keys():
                    result_dict[mk].append(df_history[mk].iloc[-1])
                else:
                    result_dict[mk] = [df_history[mk].iloc[-1]]
        return result_dict
'''