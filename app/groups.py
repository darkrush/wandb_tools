import click
from wandb_tools.core import Wandb_Local, build_siblings, find_all_groups

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option('--enterpoint',
              default='shlab_qiujiantao/VDS_NN_Formal',
              help='enterpoint of wandb')
@click.option('--dbpath',
              default='/home/qiujiantao/project/wandb_tools/cache_database',
              help='cache db dir')
def find_groups(enterpoint, dbpath):
    wandb_local = Wandb_Local(enterpoint, dbpath)
    siblings_dict = build_siblings(wandb_local)

    group_list_dict, uncoverd_list = find_all_groups(siblings_dict)
    for main_key, group_list in group_list_dict.items():
        print('========={} start========='.format(main_key))
        for group in group_list:
            print(group)
        print('========={} end========='.format(main_key))
    print(uncoverd_list)


if __name__ == '__main__':
    find_groups()
