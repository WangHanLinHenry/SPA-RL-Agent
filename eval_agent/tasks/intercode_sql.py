import os
import json
import logging
import pandas as pd
from typing import Tuple, Any, Iterable, Dict, List

logger = logging.getLogger("agent_frame")

from eval_agent.tasks.base import Task

from intercode.envs import SqlEnv


def preprocess_sql(record: Dict) -> str:
    # logger.info(record)
    db = record['extra']["db"]
    
    return f"use {db}"


class IntercodeSQLTask(Task):
    task_name = "intercode_sql"

    def __init__(
        self,
        idx: Any,
        env: SqlEnv,
        obs: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.task_id = idx
        self.env = env
        self.observation = obs
    
    @classmethod
    def load_tasks(cls, split: str, part_num: int, part_idx: int = -1) -> Tuple[Iterable[Task], int]:
        if split == 'train':
            idxs = json.load(open("eval_agent/data/intercode_sql/train_indices.json"))
            env = SqlEnv("docker-env-sql", data_path="eval_agent/data/intercode_sql/ic_spider_train.json", preprocess=preprocess_sql, verbose=False)
        else:
            idxs = json.load(open("eval_agent/data/intercode_sql/test_indices.json"))
            env = SqlEnv("docker-env-sql", data_path="eval_agent/data/intercode_sql/ic_spider_test.json", preprocess=preprocess_sql, verbose=False)
        
        if part_num == 1:
            idxs = idxs
        else:
            assert part_idx != -1
            part_len = len(idxs) // part_num + 1
            idxs = idxs[part_len * part_idx: min(part_len * (part_idx + 1), len(idxs))]
        N_TASKS = len(idxs)
        
        def generator():
            for idx in idxs:
                env.reset(idx)
                obs = env.query
                yield cls(idx, env, obs)

        return generator(), N_TASKS
    