import collections

from docplex.cp.model import *


def checkerSat(solution, instance_path):
    # Create the model.
    mdl = CpoModel()

    jobs_data = []
    machines_count = 0
    jobs_count = 0

    instance_file = open(instance_path, 'r')
    line_str = instance_file.readline()
    line_cnt = 1
    while line_str:
        data = []
        split_data = line_str.split()
        if line_cnt == 1:
            jobs_count, machines_count = int(split_data[0]), int(split_data[1])
        else:
            i = 0
            while i < len(split_data):
                machine, time = int(split_data[i]), int(split_data[i + 1])
                data.append((machine, time))
                i += 2
            jobs_data.append(data)
        line_str = instance_file.readline()
        line_cnt += 1
    instance_file.close()

    all_machines = range(machines_count)

    # Creates job intervals and add to the corresponding machine lists.
    all_tasks = {}
    machine_to_intervals = collections.defaultdict(list)

    for job_id, job in enumerate(jobs_data):
        for task_id, task in enumerate(job):
            machine = task[0]
            duration = task[1]
            suffix = "_%i_%i" % (job_id, task_id)

            interval_variable = interval_var(size=duration, name="interval" + suffix)
            machine_to_intervals[machine].append(interval_variable)
            all_tasks[job_id, task_id] = interval_variable
            mdl.add(start_of(interval_variable) == solution[job_id][task_id])

        # Create and add disjunctive constraints.
    for machine in all_machines:
        mdl.add(no_overlap(machine_to_intervals[machine]))

        # Precedences inside a job.
    for job_id, job in enumerate(jobs_data):
        for task_id in range(1, len(job)):
            mdl.add(end_before_start(all_tasks[job_id, task_id - 1], all_tasks[job_id, task_id]))

    mdl.add(minimize(max(end_of(task_interval) for task_interval in all_tasks.values())))

    # Solve model.
    res = mdl.solve(Workers=1, LogVerbosity='Quiet')
    correct_solution = res.is_solution()
    assert correct_solution, "No solution found"
    return correct_solution, res.get_objective_value()
