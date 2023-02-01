import click
from wandb_tools.core import Wandb_Local, build_siblings

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
@click.option('--sibs',
              default=None,
              help='sib to find')
def get_siblings(enterpoint, dbpath, sibs):
    wandb_local = Wandb_Local(enterpoint, dbpath)
    siblings_dict = build_siblings(wandb_local)
    if sibs is None:
        print('Found {} in project'.format(len(siblings_dict)))
        for idx, hashcode in enumerate(sorted(siblings_dict.keys())):
            print('*********{}: {} start*********'.format(idx, hashcode))
            print(siblings_dict[hashcode])
            print('*********{}: {} end*********'.format(idx, hashcode))
            print()
    else:
        sibs_hashcode_list = [code for code in siblings_dict.keys() if sibs in code]
        
        print('Found {} in project'.format(len(sibs_hashcode_list)))
        for idx, hashcode in enumerate(sorted(sibs_hashcode_list)):
            print('*********{}: {} start*********'.format(idx, hashcode))
            print(siblings_dict[hashcode])
            print('*********{}: {} end*********'.format(idx, hashcode))
            print()


if __name__ == '__main__':
    get_siblings()
