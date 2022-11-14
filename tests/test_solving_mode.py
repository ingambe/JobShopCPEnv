import json

import numpy as np

from src.compiled_jss.CPEnv import CompiledJssEnvCP
from tests.cp_checker import checkerSat


def test_ta01_solving_mode():
    env = CompiledJssEnvCP('../src/compiled_jss/instances/ta01', solving_mode=True)
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
    correct_solution, cp_makespan = checkerSat(solution_agent, '../src/compiled_jss/instances/ta01')
    assert correct_solution, f'agent solution is not correct'
    assert makespan_agent == cp_makespan, f'makespan computation is not correct'


def test_ta80_solving_mode():
    env = CompiledJssEnvCP('../src/compiled_jss/instances/ta80', solving_mode=True)
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
    correct_solution, cp_makespan = checkerSat(solution_agent, '../src/compiled_jss/instances/ta80')
    assert correct_solution, f'agent solution is not correct'
    assert makespan_agent == cp_makespan, f'makespan computation is not correct'


def test_dmu01_solving_mode():
    env = CompiledJssEnvCP('../src/compiled_jss/instances/dmu01.txt', solving_mode=True)
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
    correct_solution, cp_makespan = checkerSat(solution_agent, '../src/compiled_jss/instances/dmu01.txt')
    assert correct_solution, f'agent solution is not correct'
    assert makespan_agent == cp_makespan, f'makespan computation is not correct'


