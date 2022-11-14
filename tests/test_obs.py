import json

import numpy as np

from src.compiled_jss.CPEnv import CompiledJssEnvCP
from tests.cp_checker import checkerSat


def test_obs_ta01():
    env = CompiledJssEnvCP('../src/compiled_jss/instances/ta01')
    obs = env.reset()
    done = False
    info = {}
    while not done:
        assert 'interval_rep' in obs, f'interval_rep not in obs'
        assert 'action_mask' in obs, f'action_mask not in obs'
        assert 'index_interval' in obs, f'index_interval not in obs'
        assert 'job_resource_mask' in obs, f'job_resource_mask not in obs'
        assert 'attention_interval_mask' in obs, f'attention_interval_mask not in obs'
        mask = obs['action_mask']
        assert sum(mask) > 0, f'no action possible'
        # sample without replacement from obs['action_mask'] boolean vector
        action = np.random.choice(np.arange(len(obs['action_mask'])), size=int(sum(obs['action_mask'])),
                                  p=obs['action_mask'] / obs['action_mask'].sum(), replace=False)
        obs, reward, done, info = env.step(action)
    assert "makespan" in info, f'finished without makespan'
    makespan_agent = int(info["makespan"])
    solution_agent = json.loads(info["solution"])
    correct_solution, cp_makespan = checkerSat(solution_agent, '../src/compiled_jss/instances/ta01')
    assert correct_solution, f'agent solution is not correct'
    assert makespan_agent == cp_makespan, f'makespan computation is not correct'
