from typing import List
import pickle
import wandb
import json
import os
import logging
import shutil
from wandb import sdk as wandb_sdk

logger = logging.getLogger(__name__)
api = wandb.Api()


class Wandb_Local:
    def __init__(self, enterpoint: str, db_path: str) -> None:
        self._enterpoint = enterpoint
        self._db_path = db_path

    def clean_cache(self):
        shutil.rmtree(self._db_path)
        os.makedirs(self._db_path)

    def pull_runs(self, clean_cache: bool = False):
        runs = api.runs(self._enterpoint, per_page=100000)
        runs: List[wandb_sdk.wandb_run.Run]

        if not os.path.exists(self._db_path):
            os.makedirs(self._db_path)

        for run in runs:
            run_id = run.id

            local_run_dir = os.path.join(self._db_path, run_id)

            if not os.path.exists(local_run_dir):
                logger.info("{} already cached in local".format(run_id))
                continue
            else:
                logger.info("{} not found in local, make new dir".format(run_id))
                os.makedirs(local_run_dir)

                config = run.config
                with open(os.path.join(local_run_dir, 'config.json'), 'w') as f:
                    json.dump(config, f)

                df_history = run.history()
                feather_path = os.path.join(local_run_dir, 'history.pkl')
                with open(feather_path, 'wb') as f:
                    pickle.dump(df_history, f)
