import json
from datetime import timedelta

import numpy as np
from hypothesis import given, settings
from hypothesis.strategies import integers

from src.compiled_jss.CPEnv import CompiledJssEnvCP
from tests.cp_checker import checkerSat


def test_tai_small():
    env = CompiledJssEnvCP('../src/compiled_jss/instances/tai_j10_m10_1.data')
    obs = env.reset()
    done = False
    info = {}
    while not done:
        # sample without replacement from obs['action_mask'] boolean vector
        action = np.random.choice(np.arange(len(obs['action_mask'])), size=int(sum(obs['action_mask'])),
                                  p=obs['action_mask'] / obs['action_mask'].sum(), replace=False)
        obs, reward, done, info = env.step(action)
    assert "makespan" in info, f'finished without makespan'
    makespan_agent = int(info["makespan"])
    solution_agent = json.loads(info["solution"])
    correct_solution, cp_makespan = checkerSat(solution_agent, '../src/compiled_jss/instances/tai_j10_m10_1.data')
    assert correct_solution, f'agent solution is not correct'
    assert makespan_agent == cp_makespan, f'makespan computation is not correct'


def test_tai_large():
    env = CompiledJssEnvCP('../src/compiled_jss/instances/tai_j100_m100_1.data')
    obs = env.reset()
    done = False
    info = {}
    while not done:
        # sample without replacement from obs['action_mask'] boolean vector
        action = np.random.choice(np.arange(len(obs['action_mask'])), size=int(sum(obs['action_mask'])),
                                  p=obs['action_mask'] / obs['action_mask'].sum(), replace=False)
        obs, reward, done, info = env.step(action)
    assert "makespan" in info, f'finished without makespan'
    makespan_agent = int(info["makespan"])
    solution_agent = json.loads(info["solution"])
    correct_solution, cp_makespan = checkerSat(solution_agent, '../src/compiled_jss/instances/tai_j100_m100_1.data')
    assert correct_solution, f'agent solution is not correct'
    assert makespan_agent == cp_makespan, f'makespan computation is not correct'

@settings(deadline=None)
@given(integers(min_value=0, max_value=9))
def test_ta_random(idx):
    env = CompiledJssEnvCP(f'../src/compiled_jss/instances/ta1{idx}')
    obs = env.reset()
    done = False
    info = {}
    while not done:
        # sample without replacement from obs['action_mask'] boolean vector
        action = np.random.choice(np.arange(len(obs['action_mask'])), size=int(sum(obs['action_mask'])),
                                  p=obs['action_mask'] / obs['action_mask'].sum(), replace=False)
        obs, reward, done, info = env.step(action)
    assert "makespan" in info, f'finished without makespan'
    makespan_agent = int(info["makespan"])
    solution_agent = json.loads(info["solution"])
    correct_solution, cp_makespan = checkerSat(solution_agent,
                                               f'../src/compiled_jss/instances/ta1{idx}')
    assert correct_solution, f'agent solution is not correct'
    assert makespan_agent == cp_makespan, f'makespan computation is not correct'
