import click
from wandb_tools.core import (Wandb_Local, build_siblings, find_all_groups,
                              CARED_KEY_LIST)
from wandb_tools.utils import diff_configs
import pandas as pd
from tabulate import tabulate
from wandb_tools.table_utils import from_with_std, calc_precision
from typing import Dict, List, Any
import numpy as np
import logging
import textwrap
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option('--enterpoint',
              default='shlab_qiujiantao/VDS_NN_Formal',
              help='enterpoint of wandb')
@click.option('--dbpath',
              default='/home/qiujiantao/project/wandb_tools/cache_database',
              help='cache db dir')
@click.option('--key',
              default='input_noise',
              help='cache db dir')
def find_groups(enterpoint: str, dbpath: str, key: str):
    metric_keys = ['loop_test/mean_dis_error', 'loop_test/mean_vel_error',
                   'loop_test/mean_yaw_error']
    wandb_local = Wandb_Local(enterpoint, dbpath)
    siblings_dict = build_siblings(wandb_local)

    group_list_dict, uncoverd_list = find_all_groups(siblings_dict)
    group_list = group_list_dict[key]
    print('========={} start========='.format(key))
    config_list = [siblings_dict[group.list_sibs_hash()[0]]._config
                   for group in group_list]
    diff_keys = diff_configs(config_list, CARED_KEY_LIST)
    if key in diff_keys:
        diff_keys.remove(key)
    highlight_keys = [key,] + diff_keys
    print(highlight_keys)
    tabulate_keys = highlight_keys + metric_keys
    for group in group_list:
        raw_data_list = []
        for sib_hash in group.list_sibs_hash():
            siblings = siblings_dict[sib_hash]
            row_dict = {k: siblings._config[k] for k in highlight_keys}
            metric_dict_list: List[Dict[str, Any]] = []
            for runID in siblings._runID_list:
                history_df = wandb_local.get_history(runID)
                run_value_dict = {k: float(history_df[k].iloc[-1]) for k in metric_keys}
                metric_dict_list.append(run_value_dict)
            for k in metric_keys:
                value_vector = np.array([m[k] for m in metric_dict_list])
                valid_num = np.count_nonzero(~np.isnan(value_vector))
                run_num = len(value_vector)
                raw_element = {'mean': np.nanmean(value_vector),
                               'std': np.nanstd(value_vector, ddof=1),
                               'valid_num': valid_num,
                               'run_num': run_num}
                row_dict[k] = raw_element
            raw_data_list.append(row_dict)
        raw_data_list.sort(key=lambda x: x[key])
        raw_df = pd.DataFrame(raw_data_list, columns=tabulate_keys)

        for k in metric_keys:
            raw_column = list(raw_df[k])
            min_std = min([ele['std'] for ele in raw_column])

            def format_raw(raw_data: dict):
                str1 = from_with_std(raw_data['mean'], raw_data['std'],
                                     width=14, precision=calc_precision(min_std))
                str2 = '{}/{}'.format(raw_data['valid_num'], raw_data['run_num'])
                return '{} {}'.format(str1, str2)
            raw_df[k] = raw_df[k].apply(format_raw)

        for k in highlight_keys+metric_keys:
            raw_column = list(raw_df[k])
            raw_df[k] = raw_df[k].apply(lambda x: textwrap.fill(x, width=12))
        raw_df.columns = [textwrap.fill(k, width=12) for k in raw_df.columns]

        print(tabulate(raw_df, headers="keys", tablefmt="grid"))


if __name__ == '__main__':
    find_groups()