# Constraint Programming Based Job-Shop Scheduling Environment

This environment is designed to enable the training of Reinforcement Learning (RL) agents for solving Job-Shop Scheduling (JSS) problems using Constraint Programming (CP). 
It is fast and scalable, with a focus on end-to-end training. 
The environment provides the raw `IntervalVariable` representation as observations, with no pre-defined reward function.

We recommend the reader to check the paper **An End-to-End Reinforcement Learning Approach for Job-Shop Scheduling Problems Based on Constraint Programming** for more information.

If you need to define a reward function or a different observation, you can do so by forking the environment and modifying the `step` function.

# Installation

To install the environment, simply run the following command in your terminal:

```bash
pip install job-shop-cp-env
```

If you wish to modify the environment and compile it yourself with MyPyC, you can do so using the `setup.py` script.

# Usage

In this example we load the instance `ta80` and randomly sample actions from the action mask until the episode is done.
The `action_mask` is a boolean vector of size `n_jobs + 1` where `True` values indicate that the corresponding action is valid.
The last action corresponds to the `No-Op` action allowing to jump to the next timestep without any allocation.


```python
import json
import numpy as np
from src.compiled_jss.CPEnv import CompiledJssEnvCP

env = CompiledJssEnvCP('src/compiled_jss/instances/ta80')
obs = env.reset()
done = False
info = {}
while not done:
    # sample without replacement from obs['action_mask'] boolean vector
    action = np.random.choice(np.arange(len(obs['action_mask'])), size=int(sum(obs['action_mask'])),
                                  p=obs['action_mask'] / obs['action_mask'].sum(), replace=False)
    obs, reward, done, info = env.step(action)
makespan_agent = int(info["makespan"])
solution_agent = json.loads(info["solution"])
```

You can either pass one action at the time or a list of actions to the `step` function.
For more information, please refer to the paper.

# Implementation details

The environment implementation is defined in `src/compiled_jss/CpEnv.py` and compiled using MyPyC.
Inside, CP models variables and constraints are defined.
The CP model can be transformed into a `String` representation that can be used by `IBM CP Optimizer` to solve the problem.

# Citation

If you use this environment in your research, please cite the following paper:

```bibtex
TO BE PUBLISHED
```