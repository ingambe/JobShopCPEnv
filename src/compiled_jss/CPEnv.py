from __future__ import annotations

import collections

import sys
import json
import gym
import numpy as np

from typing import List, Dict, Tuple, ClassVar, Final
import numpy.typing as npt

from docplex.cp import modeler
from docplex.cp.expression import CpoFunctionCall, CpoIntervalVar
from docplex.cp.model import CpoModel, context
from docplex.cp.modeler import end_of, minimize, less_or_equal, start_of, end_before_start
from docplex.cp.solution import CpoSolveResult, CpoIntervalVarSolution

INTERVAL_MIN: Final = -2 ** 30
INTERVAL_MAX: Final = 2 ** 30
WINDOW_INTERVAL_SIZE: Final = 5


class IntVar:
    count_var: ClassVar[int] = 0
    _model: Model
    var_lb: int
    var_ub: int
    var_initial_lb: int
    var_initial_ub: int
    _name: str
    _associated_constraint: List[str]

    def __init__(self, model: Model, lb: int = INTERVAL_MIN, ub: int = INTERVAL_MAX, name: str = ''):
        super(IntVar, self).__init__()
        self._model = model
        self.var_lb = lb
        self.var_ub = ub
        self.var_initial_lb = lb
        self.var_initial_ub = ub
        self._associated_constraint = []
        self._name = name
        if name == '':
            self._name = f'interval_var_{self.count_var}'
        IntVar.count_var += 1

    @property
    def lb(self) -> int:
        return self.var_lb

    @lb.setter
    def lb(self, value: int) -> None:
        self.var_lb = value

    @property
    def ub(self) -> int:
        return self.var_ub

    @ub.setter
    def ub(self, value: int) -> None:
        self.var_ub = value

    @property
    def is_fixed(self) -> bool:
        return self.lb == self.ub

    def add_constraint(self, name: str) -> None:
        self._associated_constraint.append(name)

    def ask_for_propagation_on_constraint(self) -> None:
        for constraint_name in self._associated_constraint:
            if constraint_name in self._model.constraints:
                self._model.constraints[constraint_name].ask_for_propagation()
            else:
                self._associated_constraint.remove(constraint_name)

    def fix(self, value: int) -> int:
        reduced = self.lb != value or self.ub != value
        self.lb = value
        self.ub = value
        if reduced:
            self.ask_for_propagation_on_constraint()
        return reduced

    def reduce_lb(self, value: int) -> int:
        reduced = value > self.lb
        self.lb = max(self.lb, value)
        if reduced:
            self.ask_for_propagation_on_constraint()
        return reduced

    def reduce_ub(self, value: int) -> int:
        reduced = value < self.ub
        self.ub = min(self.ub, value)
        if reduced:
            self.ask_for_propagation_on_constraint()
        return reduced

    @property
    def initial_lb(self) -> int:
        return self.var_initial_lb

    @property
    def initial_ub(self) -> int:
        return self.var_initial_ub

    def __repr__(self) -> str:
        return self._name

    @property
    def name(self) -> str:
        return self._name


class IntVarOffset(IntVar):
    var: IntVar
    offset: int
    _name: str

    def __init__(self, var: IntVar, offset: int):
        super(IntVar, self).__init__()
        self.var = var
        self.offset = offset
        self._name = var.name + '_offset_' + str(offset)

    @property
    def lb(self) -> int:
        return self.var.lb + self.offset

    @lb.setter
    def lb(self, value: int) -> None:
        self.var.lb = value - self.offset

    @property
    def ub(self) -> int:
        return self.var.ub + self.offset

    @ub.setter
    def ub(self, value: int) -> None:
        self.var.ub = value - self.offset

    @property
    def is_fixed(self) -> bool:
        return self.var.is_fixed

    def add_constraint(self, name: str) -> None:
        self.var.add_constraint(name)


class IntervalVar(IntVar):
    _start: IntVar
    _end: IntVarOffset
    duration: int
    _name: str
    _model: Model

    def __init__(self, model: Model, duration: int, name: str):
        super(IntervalVar, self).__init__(model)
        self._start = IntVar(model, lb=0, name=name)
        self.duration = duration
        self._end = IntVarOffset(self._start, duration)
        self._name = name

    @property
    def start(self) -> IntVar:
        return self._start

    @property
    def end(self) -> IntVar:
        return self._end

    @property
    def is_fixed(self) -> bool:
        return self.start.is_fixed

    def __repr__(self) -> str:
        rep = f'{self.name} = intervalVar(size={self.duration}); \n'
        if self.is_fixed:
            rep += f'startOf({self.name}) == {self.start.lb};\n'
        else:
            rep += f'startOf({self.name}) >= {self.start.lb};\n'
            rep += f'startOf({self.name}) <= {self.start.ub};\n'
        return rep

    def add_constraint(self, name: str) -> None:
        self.start.add_constraint(name)
        self.end.add_constraint(name)


class Constraint:
    _name: str
    _to_propagate: bool

    def __init__(self, name: str = ''):
        self._name = name
        self._to_propagate = True

    @property
    def name(self) -> str:
        return self._name

    def propagate(self) -> int:
        raise NotImplementedError()

    @property
    def to_delete(self) -> bool:
        raise NotImplementedError()

    @property
    def to_propagate(self) -> bool:
        return self._to_propagate

    def ask_for_propagation(self) -> None:
        self._to_propagate = True


class NoOverlapConstraint(Constraint):
    count_sequence: ClassVar[int] = 0
    intervals: List[IntervalVar]

    def __init__(self, intervals: List[IntervalVar]):
        super().__init__(f'_SEQ_{NoOverlapConstraint.count_sequence}')
        self.intervals = intervals
        for interval_var in intervals:
            interval_var.add_constraint(self.name)
        NoOverlapConstraint.count_sequence += 1

    def propagate(self) -> int:
        reduced = 0
        list_fixed_intervals: List[IntervalVar] = []
        min_start_lb_non_fixed: int = INTERVAL_MAX
        for idx_interval, interval in enumerate(self.intervals):
            if interval.is_fixed:
                list_fixed_intervals.append(interval)
                start, end = interval.start.lb, interval.end.lb
                for idx_other, other in enumerate(self.intervals):
                    if idx_interval != idx_other and not other.is_fixed:
                        if other.start.lb < start < other.end.lb:
                            reduced = other.start.reduce_lb(end) + reduced
                        if start <= other.start.lb < end:
                            reduced = other.start.reduce_lb(end) + reduced
            else:
                min_start_lb_non_fixed = min(min_start_lb_non_fixed, interval.start.lb)
        for interval in list_fixed_intervals:
            if interval.end.lb <= min_start_lb_non_fixed:
                self.intervals.remove(interval)
        self._to_propagate = False
        return reduced

    def __repr__(self) -> str:
        rep = f'{self.name} = sequenceVar([{", ".join([str(interval.name) for interval in self.intervals])}]);\n'
        rep += f'noOverlap({self.name});\n'
        return rep

    @property
    def to_delete(self) -> bool:
        return False


class EndBeforeStartConstraint(Constraint):
    count_constraint: ClassVar[int] = 0
    interval1: IntervalVar
    interval2: IntervalVar

    def __init__(self, interval1: IntervalVar, interval2: IntervalVar, name: str = ''):
        if name == '':
            name = f'_END_BEFORE_START_{EndBeforeStartConstraint.count_constraint}'
        super().__init__(name)
        self.interval1 = interval1
        self.interval2 = interval2
        self.interval1.add_constraint(self.name)
        self.interval2.add_constraint(self.name)
        EndBeforeStartConstraint.count_constraint += 1

    def propagate(self) -> int:
        self._to_propagate = False
        return self.interval2.start.reduce_lb(self.interval1.end.lb)

    @property
    def to_delete(self) -> bool:
        return self.interval1.is_fixed or self.interval2.is_fixed

    def __repr__(self) -> str:
        return f'endBeforeStart({self.interval1.name}, {self.interval2.name});\n'


class ArithmeticConstraint(Constraint):
    count_constraint: ClassVar[int] = 0
    var: IntVar
    op: str
    value: int

    def __init__(self, var: IntVar, op: str, value: int, name: str = ''):
        if name == '':
            name = f'_ARITH_{op}_{value}_{ArithmeticConstraint.count_constraint}'
        super().__init__(name)
        self.var = var
        self.op = op
        self.value = value
        self.var.add_constraint(self.name)
        ArithmeticConstraint.count_constraint += 1

    def propagate(self) -> int:
        if self.op == '<=':
            reduced = self.var.reduce_ub(self.value)
        elif self.op == '>=':
            reduced = self.var.reduce_lb(self.value)
        elif self.op == '<':
            reduced = self.var.reduce_ub(self.value - 1)
        elif self.op == '>':
            reduced = self.var.reduce_lb(self.value + 1)
        elif self.op == '==':
            reduced = self.var.fix(self.value)
        else:
            raise ValueError("Unknown operator: {}".format(self.op))
        self._to_propagate = False
        return reduced

    def __repr__(self) -> str:
        return ''

    @property
    def to_delete(self) -> bool:
        return True


class Model(object):
    _vars: Dict[str, IntervalVar]
    _constraints: Dict[str, Constraint]
    _deleted_constraints: List[Constraint]

    def __init__(self) -> None:
        super(Model, self).__init__()
        self._vars: Dict[str, IntervalVar] = dict()
        self._constraints: Dict[str, Constraint] = dict()
        self._deleted_constraints: List[Constraint] = list()

    def add(self, var: IntervalVar) -> None:
        self._vars[var.name] = var

    def add_constraint(self, constraint: Constraint) -> None:
        self._constraints[constraint.name] = constraint

    def propagate(self) -> bool:
        reduced: int = 1
        while reduced >= 1:
            reduced = 0
            to_pop: List[str] = []
            for name_constraint, constraint in self._constraints.items():
                if constraint.to_propagate:
                    domain_reduced: int = constraint.propagate()
                    if constraint.to_delete:
                        to_pop.append(name_constraint)
                    reduced = domain_reduced + reduced
            for to_remove_const in to_pop:
                self._deleted_constraints.append(self._constraints[to_remove_const])
                del self._constraints[to_remove_const]
        return not any([var.is_fixed for var in self._vars.values()])

    def __repr__(self) -> str:
        rep = ''
        for var in self._vars.values():
            rep += str(var)
        for constraint in self._constraints.values():
            rep += str(constraint)
        for constraint in self._deleted_constraints:
            rep += str(constraint)
        return rep

    @property
    def vars(self) -> Dict[str, IntervalVar]:
        return self._vars

    @property
    def constraints(self) -> Dict[str, Constraint]:
        return self._constraints


class CompiledJssEnvCP:

    def __init__(self, instance_filename: str):
        # super(JssEnvCPLazy, self).__init__()
        context.solver.local.process_start_timeout = 60
        if sys.platform.startswith("linux"):
            context.solver.local.execfile = (
                "/opt/ibm/ILOG/CPLEX_Studio221/cpoptimizer/bin/x86-64_linux/cpoptimizer"
            )
        elif sys.platform == "darwin":
            context.solver.local.execfile = (
                "/Applications/CPLEX_Studio221/cpoptimizer/bin/x86-64_osx/cpoptimizer"
            )
        self.instance_filename: str = instance_filename
        self.env_name: str = self.instance_filename.split("/")[-1]

        # initial values for variables used for instance
        self.jobs_count: int = 0
        self.machines_count: int = 0
        self.current_timestamp: int = -1
        self.jobs_data: List[List[Tuple[int, int]]] = []
        self.max_time_op: float = 0.0
        self.min_time_op: float = float('inf')
        self.op_count: int = 0
        all_op_time: List[int] = []
        self.job_op_count: List[int] = []
        self.random_seed: int = 0

        instance_file = open(self.instance_filename, "r")
        line_str: str = instance_file.readline()
        line_cnt: int = 1
        while line_str:
            data = []
            split_data = line_str.split()
            if line_cnt == 1:
                self.jobs_count, self.machines_count = int(split_data[0]), int(
                    split_data[1]
                )
            else:
                i = 0
                this_job_op_count = 0
                while i < len(split_data):
                    machine, op_time = int(split_data[i]), int(split_data[i + 1])
                    data.append((machine, op_time))
                    i += 2
                    self.op_count += 1
                    all_op_time.append(op_time)
                    self.max_time_op = max(self.max_time_op, op_time)
                    self.min_time_op = min(self.min_time_op, op_time)
                    this_job_op_count += 1
                self.jobs_data.append(data)
                self.job_op_count.append(this_job_op_count)
            line_str = instance_file.readline()
            line_cnt += 1
        instance_file.close()

        self.mean_op_time: float = np.mean(all_op_time).item()
        self.std_op_time: float = np.std(all_op_time).item() + 1e-8

        self.mean_op_count: float = np.mean(self.job_op_count).item()

        self.action_space = gym.spaces.Discrete(self.jobs_count + 1)
        self.observation_space = gym.spaces.Dict(
            {
                "interval_rep": gym.spaces.Box(
                    low=-10, high=10, shape=(self.jobs_count, WINDOW_INTERVAL_SIZE, 4), dtype=np.float32
                ),
                "index_interval": gym.spaces.Box(
                    low=-1, high=10, shape=(self.jobs_count, WINDOW_INTERVAL_SIZE), dtype=np.float32
                ),
                "action_mask": gym.spaces.Box(
                    0, 1, shape=(self.jobs_count + 1,), dtype=np.float32
                ),
                "job_resource_mask": gym.spaces.Box(
                    0, 1, shape=(self.jobs_count + 1, self.jobs_count + 1), dtype=np.float32
                ),
                "attention_interval_mask": gym.spaces.Box(
                    0, 1, shape=(self.jobs_count, WINDOW_INTERVAL_SIZE), dtype=np.float32
                ),
                "start_end_tokens": gym.spaces.Box(
                    0, 2, shape=(self.jobs_count, WINDOW_INTERVAL_SIZE), dtype=np.float32
                ),
            }
        )
        # self.mean_machine_time: float = np.mean(list(self.machine_total_time.values())).item()

        self.action_mask: npt.NDArray[np.float32] = np.zeros((self.jobs_count + 1), dtype=np.float32)

        # for solving
        self.partial_solution: List[List[int]] = []
        self.make_span: int = -1
        self.model: Model = Model()
        self.no_op_end: int = INTERVAL_MAX
        self.total_allocated_op: int = 0
        self.machine_to_intervals: collections.defaultdict[int, List[IntervalVar]] = collections.defaultdict(list)
        self.all_tasks: Dict[Tuple[int, int], str] = {}
        self.machine_to_no_overlap: Dict[int, str] = {}
        self.already_added_interval_job: collections.defaultdict[int, List[str]] = collections.defaultdict(list)
        self.all_jobs_start_time: List[int] = []

    def _normalize_observation(self, observation: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        mean: npt.NDArray[np.float32] = np.array([0.0, np.mean(observation[:, 1]), self.mean_op_time, 0.0], dtype=np.float32)
        std: npt.NDArray[np.float32] = np.array([1.0, np.std(observation[:, 1]) + 1e-8, self.std_op_time + 1e-8, 1.0], dtype=np.float32)
        observation = (observation - mean) / std
        # print(observation)
        return observation

    def _get_job_resource_mask(self) -> npt.NDArray[np.float32]:
        job_resource_mask: npt.NDArray[np.float32] = np.eye(self.jobs_count + 1, self.jobs_count + 1, dtype=np.float32)
        machine_job = collections.defaultdict(list)
        for job_id in range(self.jobs_count):
            if self.action_mask[job_id]:
                machine_needed = self.jobs_data[job_id][len(self.partial_solution[job_id])][0]
                machine_job[machine_needed].append(job_id)
        for machine in machine_job:
            jobs_of_resource = machine_job[machine]
            mask_to_make = [False for _ in range(self.jobs_count + 1)]
            mask_to_make[-1] = self.action_mask[-1]
            for job_id in jobs_of_resource:
                mask_to_make[job_id] = True
            for job_id in jobs_of_resource:
                job_resource_mask[job_id, :] = mask_to_make
        job_resource_mask[self.jobs_count] = self.action_mask
        return job_resource_mask

    def _update_internal_state(self) -> None:
        self.model.propagate()
        self.all_jobs_start_time = [
            self.model.vars[self.all_tasks[job_id, len(self.partial_solution[job_id])]].start.lb
            if len(self.jobs_data[job_id]) > len(self.partial_solution[job_id])
            else INTERVAL_MAX
            for job_id in range(self.jobs_count)
        ]

        set_min_start = sorted(set(self.all_jobs_start_time))
        self.current_timestamp = set_min_start[0]

        self.no_op_end = INTERVAL_MAX
        if len(set_min_start) > 1:
            self.no_op_end = set_min_start[1]

        self.action_mask[-1] = (self.no_op_end < INTERVAL_MAX)
        for job_id in range(self.jobs_count):
            self.action_mask[job_id] = \
                (self.all_jobs_start_time[job_id] == self.current_timestamp) and self.current_timestamp != INTERVAL_MAX

    def _provide_observation(self) -> Dict[str, npt.NDArray[np.float32]]:
        fake_node: npt.NDArray[np.float32] = np.array([0, 0, 0, 0], dtype=np.float32)

        attention_mask: npt.NDArray[np.float32] = np.zeros((self.jobs_count, WINDOW_INTERVAL_SIZE), dtype=np.float32)

        list_node_reps: List[npt.NDArray[np.float32]] = []
        index_nodes: List[npt.NDArray[np.float32]] = []
        tokens_nodes: List[npt.NDArray[np.float32]] = []
        min_op = min([len(self.partial_solution[job_id]) for job_id in range(self.jobs_count)])
        for job_id in range(self.jobs_count):
            this_job_node_rep: List[npt.NDArray[np.float32]] = []
            this_job_indexes: List[float] = []
            this_job_tokens: List[float] = []
            if len(self.partial_solution[job_id]) == 0:
                this_job_node_rep.append(fake_node)
                this_job_indexes.append(min_op - 1)
                this_job_tokens.append(1)

            for task_id in range(len(self.partial_solution[job_id]) - (1 - len(this_job_node_rep)),
                                 len(self.jobs_data[job_id])):
                var = self.model.vars[self.all_tasks[job_id, task_id]]
                legal_op = len(self.partial_solution[job_id]) == task_id and not var.is_fixed
                this_job_node_rep.append(np.array(
                    [
                        var.is_fixed,
                        var.start.lb,
                        var.duration,
                        legal_op
                    ], dtype=np.float32)
                )
                this_job_indexes.append(task_id)
                this_job_tokens.append(0)
                if len(this_job_node_rep) == WINDOW_INTERVAL_SIZE:
                    break
            k = 0
            while len(this_job_node_rep) < WINDOW_INTERVAL_SIZE:
                this_job_node_rep.append(fake_node)
                this_job_indexes.append(min_op - 1)
                this_job_tokens.append(2)
                if k != 0:
                    attention_mask[job_id, len(this_job_node_rep) - 1] = True
                k += 1
            list_node_reps.append(np.stack(this_job_node_rep))
            index_nodes.append(np.stack(this_job_indexes))
            tokens_nodes.append(np.stack(this_job_tokens))
        node_representation = np.stack(list_node_reps)
        index_representation = np.stack(index_nodes)
        start_end_tokens = np.stack(tokens_nodes)

        return {"interval_rep": node_representation,
                "action_mask": self.action_mask,
                "index_interval": index_representation - index_representation.min(initial=0),
                "attention_interval_mask": attention_mask,
                "start_end_tokens": start_end_tokens}

    # def reset(self) -> Dict[str, np.ndarray]:
    def reset(self) -> Dict[str, npt.NDArray[np.float32]]:
        self.action_mask = np.zeros((self.jobs_count + 1), dtype=np.float32)
        self.current_timestamp = 0
        self.total_allocated_op = 0
        self.partial_solution = [[] for _ in range(self.jobs_count)]
        self.model = Model()

        all_machines = range(self.machines_count)

        self.all_tasks = {}
        self.machine_to_intervals = collections.defaultdict(list)

        self.machine_to_no_overlap = {}
        self.already_added_interval_job = collections.defaultdict(list)

        for job_id, job in enumerate(self.jobs_data):
            for task_id, task in enumerate(job):
                machine = task[0]
                duration = task[1]
                suffix = f"_{job_id}_{task_id}"
                if task_id < WINDOW_INTERVAL_SIZE * 2:
                    interval_variable = IntervalVar(self.model, duration=duration, name="interval" + suffix)
                    self.model.add(interval_variable)
                    self.machine_to_intervals[machine].append(interval_variable)
                    self.already_added_interval_job[job_id].append(interval_variable.name)
                    self.all_tasks[job_id, task_id] = interval_variable.name
                else:
                    break

        for machine in all_machines:
            constraint_to_add = NoOverlapConstraint(self.machine_to_intervals[machine])
            self.model.add_constraint(constraint_to_add)
            self.machine_to_no_overlap[machine] = constraint_to_add.name

        for job_id, job in enumerate(self.jobs_data):
            for task_id in range(1, len(self.already_added_interval_job[job_id])):
                self.model.add_constraint(
                    EndBeforeStartConstraint(
                        self.model.vars[self.all_tasks[job_id, task_id - 1]],
                        self.model.vars[self.all_tasks[job_id, task_id]],
                    )
                )
        self._update_internal_state()
        obs = self._provide_observation()
        obs['job_resource_mask'] = self._get_job_resource_mask()
        obs['interval_rep'] = self._normalize_observation(obs['interval_rep'])
        return obs

    def seed(self, seed: int) -> None:
        self.random_seed = seed

    def get_start_interval(self, inter_sol: CpoIntervalVarSolution) -> int:
        return int(inter_sol.get_start())

    def solve_using_cp(self, starting_point_solution: List[List[int]] = list(), time_limit: int = 10,
                       workers: int = 1) -> Tuple[bool, List[List[int]], int]:
        for job_id in range(self.jobs_count):
            while len(self.already_added_interval_job[job_id]) < len(self.jobs_data[job_id]):
                machine, time_op = self.jobs_data[job_id][len(self.already_added_interval_job[job_id])]
                suffix = f"_{job_id}_{len(self.already_added_interval_job[job_id])}"
                assert "interval" + suffix not in self.model.vars
                interval_variable = IntervalVar(self.model, duration=time_op, name="interval" + suffix)
                self.model.add(interval_variable)
                self.machine_to_intervals[machine].append(interval_variable)
                self.all_tasks[job_id, len(self.already_added_interval_job[job_id])] = interval_variable.name
                # add end before start
                before_interval_name = self.all_tasks[job_id, len(self.already_added_interval_job[job_id]) - 1]
                before_interval = self.model.vars[before_interval_name]
                end_before_cstr = EndBeforeStartConstraint(before_interval, interval_variable)
                self.model.add_constraint(end_before_cstr)
                # add constraint no overlap
                no_overlap_cstr = self.machine_to_no_overlap[machine]
                interval_variable.add_constraint(no_overlap_cstr)
                self.already_added_interval_job[job_id].append(interval_variable.name)

        mdl_make_span = CpoModel()
        mdl_make_span.import_model_string(str(self.model))

        task_make_span = [task for task in mdl_make_span.get_all_variables() if isinstance(task, CpoIntervalVar)]

        all_tasks_make_span = {}

        for task in task_make_span:
            all_tasks_make_span[int(task.get_name().split("_")[1]), int(task.get_name().split("_")[2])] = task

        make_span: CpoFunctionCall = modeler.max(
            end_of(task) for task in task_make_span
        )
        make_span.set_name("make_span")
        mdl_make_span.add(minimize(make_span))

        if len(starting_point_solution) > 0:
            stp = mdl_make_span.create_empty_solution()
            for job_id in range(len(starting_point_solution)):
                for task_id in range(len(starting_point_solution[job_id])):
                    stp.add_interval_var_solution(all_tasks_make_span[job_id, task_id], True,
                                                  starting_point_solution[job_id][task_id],
                                                  starting_point_solution[job_id][task_id] +
                                                  self.jobs_data[job_id][task_id][1],
                                                  self.jobs_data[job_id][task_id][1])
            mdl_make_span.set_starting_point(stp)

        cp_result: CpoSolveResult = mdl_make_span.solve(LogVerbosity="Quiet", TimeLimit=time_limit, Workers=workers)
        assigned_jobs: collections.defaultdict[int, List[CpoIntervalVarSolution]] = collections.defaultdict(list)
        if cp_result and cp_result.is_solution():
            result = []
            for job_id in range(len(self.jobs_data)):
                job_result = []
                for task_id in range(len(self.jobs_data[job_id])):
                    job_result.append(
                        int(
                            cp_result.get_var_solution(
                                all_tasks_make_span[job_id, task_id]
                            ).start
                        )
                    )
                    machine = self.jobs_data[job_id][task_id][0]
                    assigned_jobs[machine].append(
                        cp_result.get_var_solution(all_tasks_make_span[job_id, task_id])
                    )
                result.append(job_result)
        else:
            assert cp_result.is_solution(), f'No solution found for the problem {self.instance_filename}'
            return False, [], INTERVAL_MAX

        mdl_compress = CpoModel()
        mdl_compress.import_model_string(str(self.model))

        task_compress: List[CpoIntervalVar] = [task for task in mdl_compress.get_all_variables() if
                                               isinstance(task, CpoIntervalVar)]

        all_tasks_compress: Dict[Tuple[int, int], CpoIntervalVar] = {}

        for task in task_compress:
            all_tasks_compress[int(task.get_name().split("_")[1]), int(task.get_name().split("_")[2])] = task

        for machine in range(self.machines_count):
            # Sort by starting time.
            assigned_jobs[machine] = sorted(
                assigned_jobs[machine], key=self.get_start_interval
            )
            for task_id in range(1, len(assigned_jobs[machine])):
                before: CpoIntervalVarSolution = assigned_jobs[machine][task_id - 1]
                before_index: List[str] = before.get_name().split("_")
                before_index_int: Tuple[int, int] = int(before_index[1]), int(before_index[2])
                after: CpoIntervalVarSolution = assigned_jobs[machine][task_id]
                after_index: List[str] = after.get_name().split("_")
                after_index_int: Tuple[int, int] = int(after_index[1]), int(after_index[2])
                mdl_compress.add(
                    end_before_start(
                        all_tasks_compress[before_index_int], all_tasks_compress[after_index_int]
                    )
                )

        for job_id in range(self.jobs_count):
            for task_id in range(0, len(result[job_id])):
                mdl_compress.add(
                    less_or_equal(
                        start_of(all_tasks_compress[job_id, task_id]),
                        result[job_id][task_id],
                    )
                )

        compress_obj: CpoFunctionCall = sum(
            start_of(task) for task in all_tasks_compress.values()
        )
        compress_obj.set_name("compress_obj")
        mdl_compress.add(minimize(compress_obj))
        cp_result = mdl_compress.solve(Workers=1, LogVerbosity="Quiet")

        objective: int = 0
        if cp_result and cp_result.is_solution():
            for job_id in range(self.jobs_count):
                for task_id in range(0, len(self.jobs_data[job_id])):
                    result[job_id][task_id] = int(cp_result.get_var_solution(all_tasks_compress[job_id, task_id]).start)
                    objective = max(objective, result[job_id][task_id] + self.jobs_data[job_id][task_id][1])
        else:
            print('BUG ON COMPRESSION')
            return False, [], INTERVAL_MAX

        return True, result, objective

    def step(self, actions: npt.NDArray[np.longlong]) -> \
            Tuple[Dict[str, npt.NDArray[np.float32]], float, bool, Dict[str, str]]:
        action_took: List[int] = []
        start_timestep = self.current_timestamp
        i = 0
        #obs: Dict[str, npt.NDArray[np.float32]] = {}
        infos: Dict[str, str] = {}
        done = False
        while i < len(actions) and self.current_timestamp == start_timestep and not done:
            job_id = actions[i].item()
            if self.action_mask[job_id] == 1:
                _, reward, done, infos = self.one_action(job_id)
                action_took.append(job_id)
            i += 1
        obs = self._provide_observation()
        obs['job_resource_mask'] = self._get_job_resource_mask()
        obs['interval_rep'] = self._normalize_observation(obs['interval_rep'])
        infos['action_took'] = json.dumps(action_took)
        return obs, 0.0, done, infos

    def one_action(self, action: int) -> Tuple[Dict[str, npt.NDArray[np.float32]], float, bool, Dict[str, str]]:
        if action >= self.jobs_count:
            for job_id in range(self.jobs_count):
                for task_id in range(len(self.partial_solution[job_id]),
                                     min(len(self.partial_solution[job_id]) + 1, len(self.jobs_data[job_id]))):
                    self.model.add_constraint(
                        ArithmeticConstraint(self.model.vars[self.all_tasks[job_id, task_id]].start, '>=',
                                             self.no_op_end)
                    )
        else:
            self.model.add_constraint(
                ArithmeticConstraint(self.model.vars[self.all_tasks[action, len(self.partial_solution[action])]].start,
                                     '==', self.current_timestamp)
            )
            self.partial_solution[action].append(self.current_timestamp)
            self.total_allocated_op += 1
            # lazy interval
            if len(self.already_added_interval_job[action]) < len(self.jobs_data[action]):
                machine, time_op = self.jobs_data[action][len(self.already_added_interval_job[action])]
                suffix = f"_{action}_{len(self.already_added_interval_job[action])}"
                interval_variable = IntervalVar(self.model, duration=time_op, name="interval" + suffix)
                self.model.add(interval_variable)
                self.machine_to_intervals[machine].append(interval_variable)
                self.all_tasks[action, len(self.already_added_interval_job[action])] = interval_variable.name
                # add end before start
                before_interval_name = self.all_tasks[action, len(self.already_added_interval_job[action]) - 1]
                before_interval = self.model.vars[before_interval_name]
                end_before_cstr = EndBeforeStartConstraint(before_interval, interval_variable)
                self.model.add_constraint(end_before_cstr)
                # add constraint no overlap
                no_overlap_cstr = self.machine_to_no_overlap[machine]
                interval_variable.add_constraint(no_overlap_cstr)
                self.already_added_interval_job[action].append(interval_variable.name)

        self._update_internal_state()
        #obs = self._provide_observation()
        is_done = self.current_timestamp == INTERVAL_MAX

        info_dict: Dict[str, str] = {
            "env_name": self.env_name,
            "env_instance": self.instance_filename,
        }

        if is_done:
            makespan = 0
            for job_id in range(self.jobs_count):
                makespan = max(makespan, self.partial_solution[job_id][-1] + self.jobs_data[job_id][-1][1])
            info_dict["makespan"] = str(makespan)
            info_dict['solution'] = json.dumps(self.partial_solution)
        return {}, 0.0, is_done, info_dict

    def render(self, mode: str = "human", png_filename: str = '', fig_size: Tuple[int, int] = (45, 50),
               font_size: int = 30) -> None:
        pass
