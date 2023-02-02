import click
from wandb_tools.core import Wandb_Local

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option('--enterpoint',
              default='shlab_qiujiantao/VDS_NN_Kinetic_v2',
              help='enterpoint of wandb')
@click.option('--dbpath',
              default='/home/qiujiantao/project/wandb_tools/cache_database',
              help='cache db dir')
@click.option('--clean_db', is_flag=True, help='clean db dir')
def get_local(enterpoint, dbpath, clean_db):
    wandb_local = Wandb_Local(enterpoint, dbpath)
    wandb_local.pull_runs(cache_clean=clean_db)


if __name__ == '__main__':
    get_local()
