# A Constraint Programming Based Job-Shop Scheduling Environment

This is a Constraint Programming (CP) based Job-Shop Scheduling (JSS) Environment that can be combined with Reinforcement-Learning (RL) to train an agent to solve a scheduling problem.
Compared to other environments, this CP based one is made to be fast and scalable.
If you install it using `setup.py`, it will automatically compile using MyPyC. 
Also, this environment has been developed for an end-to-end approach, there is no pre-defined reward function and the observation are simply the raw `IntervalVariable` representation.
We recommend the reader to check the paper for more information.

If you need to define a reward function or a different observation, you can do so by forking the environment and modifying the `step` function.

# Installation

```bash
pip install jss_cp
```