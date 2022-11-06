from src.compiled_jss.CPEnv import CompiledJssEnvCP
import numpy as np
import json

from tests.cp_checker import checkerSat

from hypothesis import given
from hypothesis.strategies import integers


def test_tai_small():
    if __name__ == '__main__':
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
    if __name__ == '__main__':
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


@given(integers(min_value=1, max_value=10))
def test_tai_random_large(idx):
    if __name__ == '__main__':
        env = CompiledJssEnvCP(f'../src/compiled_jss/instances/tai_j100_m100_{idx}.data')
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
        correct_solution, cp_makespan = checkerSat(solution_agent, f'../src/compiled_jss/instances/tai_j100_m100_{idx}.data')
        assert correct_solution, f'agent solution is not correct'
        assert makespan_agent == cp_makespan, f'makespan computation is not correct'