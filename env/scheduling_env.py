
from env.agent import *
import numpy as np
from datetime import *
import gym
import copy
from gym import spaces
from collections import deque
import random
import math
import pandas as pd

MAX_PRIORITY = 1


class SchedulingEnv(gym.Env):

    def __init__(self):
        self.info = file_to_info("Check Scheduling Input.xlsx")
        self.c_initial = self.info[1]
        self.a_initial = self.info[2]
        self.additional = self.info[6]
        self.c_not_allowed = self.info[7]
        self.a_not_allowed = self.info[10]
        self.c_elapsed_time = self.info[5]
        self.dfh = self.info[3]
        self.dfc = self.info[4]
        self.n_days = 1095
        self.n_tasks = len(self.c_initial) + len(self.a_initial)
        self.a_slots = 1
        self.c_slots = 3
        self.a_length = 1
        self.c_length = 21
        self.c_interval = 3

        self.conf_number = 0
        self.c_check_conf_number = 0
        self.a_check_conf_number = 0

        self.last_c_days = [task[4] for task in self.c_initial]
        self.last_elapsed_index = np.zeros(len(self.c_initial))
        self.last_a_days = [task[4] for task in self.a_initial]

        self.total_aircraft = len(self.c_initial)

        self.calendar = []
        self.aircraft = []
        self.task_list = []
        self.conflicts = deque()
        self.last_scheduled = 0
        self.done = False

        self.observation_space = spaces.Box(low=0, high=1, shape=(15, 100))  # Partial Calendar
        self.state = None

        self.action_space = spaces.Discrete(4)  # Number of dispatching rules

        self.more_tasks = False

        self.init_calendar()
        self.init_aircraft()
        self.create_tasks_v2()

        build_input_json(self.task_list)

        self.backup = copy.deepcopy(self.task_list)

    def reset(self):

        self.conf_number = 0
        self.c_check_conf_number = 0
        self.a_check_conf_number = 0
        self.last_scheduled = 0
        self.done = False
        self.conflicts = deque()
        self.more_tasks = False

        self.c_initial = self.info[1]
        self.a_initial = self.info[2]
        self.additional = self.info[6]
        self.c_not_allowed = self.info[7]
        self.a_not_allowed = self.info[10]
        self.last_elapsed_index = np.zeros(len(self.c_initial))
        self.task_list = copy.deepcopy(self.backup)
        self.init_aircraft()

        self.n_tasks = len(self.c_initial) + len(self.a_initial)
        self.a_slots = 1
        self.c_slots = 3
        self.a_length = 1
        self.c_length = 21
        self.c_interval = 3

        # Calendar
        self.calendar = []
        self.init_calendar()

        if not self.conflicts:  # if there are no conflicts
            # schedules every task (starting in task 0) to its due date until it finds one conflict
            while self.last_scheduled < len(self.task_list) and not self.conflicts:
                if date_to_day(self.task_list[self.last_scheduled].end_day) < self.n_days \
                        and not self.task_list[self.last_scheduled].scheduled:
                    self.schedule_maintenance(self.task_list[self.last_scheduled])
                    self.task_list[self.last_scheduled].scheduled = True
                    self.last_scheduled += 1

                if self.last_scheduled == len(self.task_list):
                    index = self.verify_tasks_scheduled()
                    if index != -1:  # if there is at least one task not scheduled
                        self.last_scheduled = index

        if not self.conflicts:
            self.done = True
            self.build_state()
        else:
            self.build_state()
        return self.state

    def step(self, action):
        self.conf_number += 1
        # TODO: Simulador devolve a reward imediatamente e o agente verifica
        # se ha um novo conflito senao nao faz nada ate haver
        assert (action < 4), "Invalid action"
        conflict = self.conflicts[-1]

        if conflict.tasks[0].type == "c-check":
            self.c_check_conf_number += 1
        else:
            self.a_check_conf_number += 1

        if action == 0:
            move = edd(conflict.tasks)
        elif action == 1:
            move = tdd(conflict.tasks)
        elif action == 2:
            move = llf(conflict.tasks)
        else:
            move = slf(conflict.tasks)

        self.solve_conflict(move)

        if not self.conflicts:  # if there are no conflicts
            # schedules every task (starting in task 0) to its due date until it finds one conflict
            while self.last_scheduled < len(self.task_list) and not self.conflicts:
                # print(self.last_scheduled)
                # print(self.task_list[self.last_scheduled].scheduled)
                if date_to_day(self.task_list[self.last_scheduled].due_date) < self.n_days \
                        and not self.task_list[self.last_scheduled].scheduled:
                    self.schedule_maintenance(self.task_list[self.last_scheduled])
                    self.task_list[self.last_scheduled].scheduled = True
                self.last_scheduled += 1

                if self.last_scheduled == len(self.task_list):
                    index = self.verify_tasks_scheduled()
                    if index != -1:  # if there is at least one task not scheduled
                        self.last_scheduled = index

        if not self.conflicts:
            self.done = True
            self.build_state()
        else:
            self.build_state()

        # reward of the state with the initial conflict
        reward = calculate_rewards(conflict.tasks)

        return self.state, reward, self.done, {}

    def schedule_maintenance(self, task, move=False):
        self.more_tasks = True

        if not move:
            task.starting_day = task.due_date
            task.end_day = task.starting_day + timedelta(task.length - 1)
            self.calculate_task_losses(task)

        task.scheduled = True
        # verify if the task can be scheduled to these days (due to possible restrictions)
        task = self.verify_availability(task)
        #print(task.id)
        #print(task.due_date)
        #print(task.starting_day)
        #print(task.end_day)
        #print("------")
        t_type = task.type
        if t_type == "a-check":
            for i in range(date_to_day(task.starting_day), date_to_day(task.end_day) + 1):
                self.calendar[i].a_checks.append(task)
        elif t_type == "c-check":  # Schedule c-check in the several days
            for i in range(date_to_day(task.starting_day), date_to_day(task.end_day) + 1):
                self.calendar[i].c_checks.append(task)

        # Checking conflicts
        self.check_conflicts(task)

        # Change due date following task
        for i in range(task.number + 1, len(self.task_list), 1):
            # use the id to verify the next tasks
            if self.task_list[i].id[:-1] == task.id[:-1]:
                self.calculate_due_date(self.task_list[i], date_to_day(task.end_day),
                                        get_task_interval(task, self.aircraft))

                # if the task was moved, then we need to reschedule the next tasks for that aircraft
                if task.starting_day < task.due_date and self.task_list[i].scheduled:
                    self.delete_conflicts(self.task_list[i])
                    self.delete_from_calendar(self.task_list[i])

                    self.task_list[i].scheduled = False
                else:
                    break

    # choose the best day to move a maintenance in case there is a conflict
    def solve_conflict(self, task):
        self.delete_conflicts(task)  # delete conflicts with this task
        # Find new day
        found = True

        previous_start_day = date_to_day(task.starting_day)
        previous_end_day = date_to_day(task.end_day)

        if task.type == "a-check":
            """
            new_day = self.find_a_day(task)
            if new_day != -1:
                task.starting_day = new_day
                task.end_day = task.starting_day + timedelta(task.length - 1)
                self.calculate_task_losses(task)
            else:
            """
            if not task.up:
                new_day = task.starting_day - timedelta(1)
                if date_to_day(new_day) >= 0:
                    task.starting_day = new_day
                    task.end_day = task.starting_day + timedelta(task.length - 1)
                    self.calculate_task_losses(task)
                else:
                    # task can not be scheduled before its due date
                    task.up = True
                    task.starting_day = task.due_date
                    self.calculate_task_losses(task)

            if task.up:
                # enters here only if the task could not be scheduled before its due date
                new_day = task.starting_day + timedelta(1)
                if new_day + timedelta(task.length) < day_to_date(self.n_days):
                    task.starting_day = new_day
                    task.end_day = task.starting_day + timedelta(task.length - 1)
                    self.calculate_task_losses(task)
                else:
                    print("not found:", task.number, date_to_day(task.due_date))
                    found = False

        elif task.type == "c-check":
            new_day = self.find_c_day(task)
            if date_to_day(new_day) >= 0 and not task.up:
                task.starting_day = new_day
                task.end_day = task.starting_day + timedelta(task.length - 1)
                self.calculate_task_losses(task)
            else:
                # enters here only if the task could not be scheduled before its due date
                task.up = True
                new_day = self.find_c_tolerance(task)
                if new_day < task.due_date + timedelta(task.tolerance) and new_day + task.length < day_to_date(
                        self.n_days):
                    task.starting_day = new_day
                    task.end_day = task.starting_day + timedelta(task.length - 1)
                    self.calculate_task_losses(task)
                else:
                    print("not found:", task.number, date_to_day(task.due_date))
                    found = False

        # self.delete_from_calendar(task)

        if task.type == "a-check":  # Erase task from calendar
            for d in range(previous_start_day, previous_end_day + 1):
                i = 0
                while i < (len(self.calendar[d].a_checks)):
                    # delete the same task in the days after
                    t = self.calendar[d].a_checks[i]
                    if t.id == task.id:
                        del self.calendar[d].a_checks[i]
                        i -= 1
                    i += 1

        elif task.type == "c-check":  # Erase task from calendar
            for d in range(previous_start_day, previous_end_day + 1):
                i = 0
                while i < (len(self.calendar[d].c_checks)):
                    # delete the same task in the days after
                    t = self.calendar[d].c_checks[i]
                    if t.id == task.id:
                        del self.calendar[d].c_checks[i]
                        i -= 1
                    i += 1
        #print(task.id)
        #print(task.due_date)
        #print(task.starting_day)
        #print(task.tolerance)
        #print("---------")
        if found:
            self.schedule_maintenance(task, move=True)
        else:
            task.starting_day = day_to_date(-1)
            task.end_day = day_to_date(-1)
            self.calculate_task_losses(task)

    # verify if all tasks are scheduled (returns the index of the first task not scheduled and -1 if all are scheduled)
    def verify_tasks_scheduled(self):
        for index in range(len(self.task_list)):
            if not self.task_list[index].scheduled:
                return index
        return -1

    # verify if the task can be scheduled to its date (due to possible restrictions)
    # returns different starting and end dates if it can't
    # the new dates correspond to the closest available slot to the left if available, if not it uses the tolerance
    def verify_availability(self, task):
        # verify if work is interrupted in any day of the task
        available = True
        for i in range(date_to_day(task.starting_day), date_to_day(task.end_day) + 1):
            if not self.calendar[i].a_check_available and task.type == "a-check":
                available = False
                break
            elif not self.calendar[i].c_check_available and task.type == "c-check":
                available = False
                break
        if available:
            return task
        else:
            task.starting_day = self.get_closest_available_day(task)
            task.end_day = task.starting_day + timedelta(task.length - 1)
            self.calculate_task_losses(task)
            return task

    def delete_from_calendar(self, task):
        if task.starting_day != -1 and task.end_day != -1:
            for d in range(date_to_day(task.starting_day), date_to_day(task.end_day) + 1):
                if task.type == "a-check":
                    for index in range(len(self.calendar[d].a_checks)):
                        if self.calendar[d].a_checks[index].id == task.id:
                            del self.calendar[d].a_checks[index]
                            break
                else:
                    for index in range(len(self.calendar[d].c_checks)):
                        if self.calendar[d].c_checks[index].id == task.id:
                            del self.calendar[d].c_checks[index]
                            break

    # delete task from conflicts and removes the entire conflict if the task solved the conflict
    def delete_conflicts(self, task):
        if task.starting_day != -1 and task.end_day != -1:
            i = 0
            while i < len(self.conflicts):  # Erase other conflicts with this task
                for k in range(len(self.conflicts[i].tasks)):
                    t = self.conflicts[i].tasks[k]
                    if t.id == task.id:
                        if t.type == "c-check" and len(self.conflicts[i].tasks) <= 4:
                            del self.conflicts[i]
                            i -= 1
                            break
                        elif t.type == "a-check" and len(self.conflicts[i].tasks) <= 2:
                            del self.conflicts[i]
                            i -= 1
                            break
                        else:
                            del self.conflicts[i].tasks[k]
                            break
                i += 1

    def insert_in_calendar(self, task):
        if task.starting_day != -1 and task.end_day != -1:
            for d in range(date_to_day(task.starting_day), date_to_day(task.end_day) + 1):
                if task.type == "a-check":
                    self.calendar[d].a_checks.append(task)
                else:
                    self.calendar[d].c_checks.append(task)

    def check_conflicts(self, task):
        start_day = date_to_day(task.starting_day)
        end_day = date_to_day(task.end_day)
        for d in range(start_day, end_day + 1):
            new_conf = Conflict([task])
            if task.type == "c-check":
                checking_day = self.calendar[d]
                if len(checking_day.c_checks) > 3:
                    for t in checking_day.c_checks:
                        # if the task is not in the conflict already, add it
                        if check_contains(t, new_conf.tasks) == -1:
                            new_conf.tasks.append(t)
            if len(new_conf.tasks) > 3:
                self.conflicts.append(new_conf)

            elif task.type == "a-check":
                if len(self.calendar[start_day].a_checks) > 1:
                    new_conf = Conflict(self.calendar[start_day].a_checks)
                    self.conflicts.append(new_conf)

    # TODO add day restrictions
    def find_a_day(self, task):
        found = False
        new_day = date_to_day(task.starting_day)
        while not found:
            new_day = new_day - 1
            if new_day < 0:
                new_day = date_to_day(task.starting_day)
                break
            else:
                if len(self.calendar[new_day].a_checks) < 1:
                    found = True
                    break

        while not found:
            # enters here if it cannot be scheduled before the due date
            new_day = new_day + 1
            if new_day < date_to_day(task.due_date) + task.tolerance and len(self.calendar[new_day].a_checks) < 1:
                found = True
                break
            else:
                found = False
                break

        if found:
            return day_to_date(new_day)
        return -1
    # finds the closest available slot to move a c-check which is right before the bigger starting_day of
    # all the tasks in the conflict
    def find_c_day(self, task):
        bigger_start_day = -99999

        for d in range(date_to_day(task.starting_day), date_to_day(task.end_day) + 1):
            conflicted = self.calendar[d].c_checks
            for c in conflicted:
                if date_to_day(c.starting_day) > bigger_start_day and task.id != c.id:
                    bigger_start_day = date_to_day(c.starting_day)

        return day_to_date(bigger_start_day - task.length)

    # finds the first available slot in the tolerance of a c-check
    def find_c_tolerance(self, task):
        due_date = date_to_day(task.due_date)
        for d in range(due_date, due_date + task.tolerance):
            if len(self.calendar[d].c_checks) < self.c_slots:
                return day_to_date(d)
        return -1

    # finds the closest day to schedule a task that meets the restrictions
    # if the task is a c-check the work is interrupted in a period of time (no less then 7 days straight), meaning
    # that the maintenance is moved to the left of that period (if available) to prevent the aircraft being
    # unavailable for larger periods of time
    def get_closest_available_day(self, task):
        found = False
        # if the task is already starting after its due date it means that it
        # cannot be scheduled before the due date, so we need to search the days after
        # if its scheduled before the due date we need to search the days before
        if task.starting_day <= task.due_date:
            scheduled_after = False
        else:
            scheduled_after = True
        day_before = date_to_day(task.starting_day)
        day_after = date_to_day(task.starting_day)
        new_day = date_to_day(task.starting_day)
        while not found:
            found = True
            if not scheduled_after:
                day_before = day_before - 1
                if day_before >= 0:
                    new_day = day_before
                else:
                    scheduled_after = True

            if scheduled_after:
                day_after = day_after + 1
                new_day = day_after

            #if new_day > date_to_day(task.due_date) + task.tolerance:
            #    print("Task is scheduled after tolerance!")

            for i in range(new_day, new_day + task.length):
                if not self.calendar[i].a_check_available and task.type == "a-check":
                    found = False
                    break
                if not self.calendar[i].c_check_available and task.type == "c-check":
                    found = False
                    break

        return day_to_date(new_day)

    def init_aircraft(self):
        self.aircraft = []
        for a in range(self.total_aircraft):
            fleet = self.c_initial[a][0]
            tail_number = self.c_initial[a][1]
            a_check_interval_dy = self.a_initial[a][7]
            a_check_interval_fh = self.a_initial[a][8]
            a_check_interval_fc = self.a_initial[a][9]
            c_check_interval_dy = self.c_initial[a][7]
            c_check_interval_fh = self.c_initial[a][8]
            c_check_interval_fc = self.c_initial[a][9]
            dfh = []
            dfc = []
            for index in range(2, len(self.dfh[a])):
                dfh.append(self.dfh[a][index])
                dfc.append(self.dfc[a][index])
            self.aircraft.append(Aircraft(fleet, tail_number, a_check_interval_dy, a_check_interval_fh,
                                          a_check_interval_fc, c_check_interval_dy, c_check_interval_fh,
                                          c_check_interval_fc, dfh, dfc))

    def init_calendar(self):
        # get day restrictions
        a_check_restrictions = []
        c_check_restrictions = []
        #a_check_restrictions = [pd.to_datetime(d[0]).date() for d in self.a_not_allowed]
        #c_check_restrictions = [[pd.to_datetime(d[0]).date(), pd.to_datetime(d[1]).date()] for d in self.c_not_allowed]
        for i in range(self.n_days):
            day = Day(START_DATE + timedelta(days=i))
            if START_DATE + timedelta(i) in a_check_restrictions:
                day.a_check_available = False
            for interval in c_check_restrictions:
                if interval[0] <= START_DATE + timedelta(i) <= interval[1]:
                    day.c_check_available = False
            self.calendar.append(day)

    def create_tasks(self):
        self.task_list = []
        task_number = 0
        a_checks = 0
        c_checks = 0

        # C-checks
        # iterates over each aircraft
        for index in range(len(self.c_initial)):
            first = True
            task_info = self.c_initial[index]
            # iterates over the length values of 5 future c-checks for each aircraft
            for c_check_index in range(2, len(self.c_elapsed_time[0])):
                t_length = self.c_elapsed_time[index][c_check_index]
                # if length == -1 then the aircraft is phased out
                if t_length == -1:
                    t_length = 15

                # define random priority
                task_priority = random.randint(1, MAX_PRIORITY)
                new_task = Task(str(task_info[1]) + "c" + str(c_check_index - 2), "c-check", t_length, task_info[10],
                                task_info[4], task_info[5], task_info[6], task_number, task_info[0], task_info[1], task_priority)
                if first:
                    new_task = self.setup_initial_task(new_task, True)
                    first = False
                else:
                    new_task = self.setup_initial_task(new_task, False)
                # If maintenance already passed its due date (rare) (case when t = 47)
                if date_to_day(new_task.due_date) < 0:
                    new_task.due_date = day_to_date(0)
                    new_task.starting_day = new_task.due_date
                    new_task.end_day = new_task.starting_day + timedelta(new_task.length - 1)
                    self.calculate_task_losses(new_task)

                # If due date is less than the number of days the task is valid
                if date_to_day(new_task.end_day) < self.n_days:
                    self.task_list.append(new_task)  # add new task to the global task list
                    if not check_aircraft(self.aircraft, new_task):
                        # add new task to the aircraft task list
                        self.add_task_to_aircraft(new_task)
                    c_checks += 1
                    task_number += 1

        # A-checks
        for index in range(len(self.a_initial)):
            first = True
            task_info = self.a_initial[index]
            t_length = 1
            current_day = 0
            count = 0
            while current_day < self.n_days:
                task_priority = random.randint(1, MAX_PRIORITY)
                new_task = Task(str(task_info[1]) + "a" + str(count), "a-check", self.a_length, task_info[10],
                                task_info[4], task_info[5], task_info[6], task_number, task_info[0], task_info[1], task_priority)
                if first:
                    new_task = self.setup_initial_task(new_task, True)
                    first = False
                else:
                    new_task = self.setup_initial_task(new_task, False)
                # If maintenance already passed its due date (rare)
                if date_to_day(new_task.due_date) < 0:
                    new_task.due_date = day_to_date(0)
                    new_task.starting_day = new_task.due_date
                    new_task.end_day = new_task.starting_day + timedelta(new_task.length - 1)
                    self.calculate_task_losses(new_task)

                # If due date is less than the number of days
                if date_to_day(new_task.end_day) < self.n_days:
                    self.task_list.append(new_task)  # add new task to the global task list
                    if not check_aircraft(self.aircraft, new_task):
                        # add new task to the respective aircraft task list
                        self.add_task_to_aircraft(new_task)

                    a_checks += 1
                    task_number += 1
                    count += 1
                current_day = date_to_day(new_task.end_day)
        print("A-checks: ", a_checks, "   C-checks: ", c_checks, "   tasks: ", len(self.task_list))

    def create_tasks_v2(self):
        self.task_list = []

        task_number = 0
        a_checks = 0
        c_checks = 0
        index = 0
        length_index = 0

        while index < self.n_days:
            more_tasks = False
            # C-checks
            for t in range(len(self.c_initial)):
                task_info = self.c_initial[t]
                if length_index < len(self.c_elapsed_time[0]) - 2:
                    t_length = self.c_elapsed_time[t][length_index + 2]

                    if t_length == -1:
                        t_length = 15
                    # define random priority
                    task_priority = random.randint(1, MAX_PRIORITY)
                    new_task = Task(str(task_info[1]) + "c" + str(index), "c-check", t_length, task_info[10],
                                    task_info[4], task_info[5], task_info[6], task_number, task_info[0], task_info[1], task_priority)

                    if index == 0:
                        new_task = self.setup_initial_task(new_task, True)
                    else:
                        new_task = self.setup_initial_task(new_task, False)
                    # If maintenance already passed its due date its gonna be less than zero (rare)
                    if date_to_day(new_task.due_date) < 0:
                        new_task.due_date = day_to_date(0)
                        new_task.starting_day = new_task.due_date
                        new_task.end_day = new_task.starting_day + timedelta(new_task.length - 1)
                        self.calculate_task_losses(new_task)
                    # If due date is less than the number of days
                    if date_to_day(new_task.end_day) < self.n_days:
                        self.task_list.append(new_task)
                        if not check_aircraft(self.aircraft, new_task):
                            # add new task to the aircraft task list
                            self.add_task_to_aircraft(new_task)
                        c_checks += 1
                        more_tasks = True
                        task_number += 1

            # A-checks
            for t in range(len(self.a_initial)):
                task_info = self.a_initial[t]
                t_length = self.a_length
                # define random priority
                task_priority = random.randint(1, MAX_PRIORITY)
                new_task = Task(str(task_info[1]) + "a" + str(index), "a-check", t_length, task_info[10],
                                task_info[4], task_info[5], task_info[6], task_number, task_info[0], task_info[1], task_priority)

                if index == 0:
                    new_task = self.setup_initial_task(new_task, True)
                else:
                    new_task = self.setup_initial_task(new_task, False)
                # If maintenance already passed its due date its gonna be less than zero (rare) (case when t = 47)
                if date_to_day(new_task.due_date) < 0:
                    new_task.due_date = day_to_date(0)
                    new_task.starting_day = new_task.due_date
                    new_task.end_day = new_task.starting_day + timedelta(new_task.length - 1)
                    self.calculate_task_losses(new_task)
                # If due date is less than the number of days
                if date_to_day(new_task.end_day) < self.n_days:
                    self.task_list.append(new_task)
                    if not check_aircraft(self.aircraft, new_task):
                        # add new task to the respective aircraft task list
                        self.add_task_to_aircraft(new_task)

                    a_checks += 1
                    more_tasks = True
                    task_number += 1

            if not more_tasks:
                break

            index += 1
            length_index += 1

        print("a-checks: ", a_checks, "   c-checks: ", c_checks, "   tasks: ", len(self.task_list))

    def build_state(self):
        self.state = np.zeros((15, 100))
        if not self.conflicts:
            return
        conf = self.conflicts[-1]
        conf_tasks = conf.tasks

        # Save bigger and smaller day
        smaller = conf_tasks[0]
        bigger = conf_tasks[0]
        for t in conf_tasks:
            if t.starting_day < smaller.starting_day:
                smaller = t
            if t.starting_day > bigger.starting_day:
                bigger = t

        # Select a number of days before and after the conflict
        left_bound = date_to_day(smaller.starting_day) - self.c_length
        right_bound = date_to_day(bigger.starting_day) + self.c_length

        empty_spaces = 100 - (right_bound - left_bound)

        left_bound -= int(empty_spaces / 2)
        right_bound += int(empty_spaces / 2)

        days = []
        for t in conf_tasks:
            days.append(date_to_day(t.starting_day))

        if left_bound < 0:
            right_bound += 0 - left_bound
            left_bound = 0
        if right_bound > self.n_days:
            left_bound -= right_bound - self.n_days
            right_bound = self.n_days

        # Build State
        diff = 0
        for d in range(left_bound, right_bound):
            day = self.calendar[d]

            if conf_tasks[0].type == "c-check":
                checks = sorted(day.c_checks, key=lambda x: x.number)
                for t in range(len(checks)):
                    task = checks[t]
                    index = check_contains(task, conf_tasks)
                    if index != -1:
                        self.state[index][diff] = 2
            elif conf_tasks[0].type == "a-check":
                checks = sorted(day.a_checks, key=lambda x: x.number)
                for t in range(len(checks)):
                    task = checks[t]
                    index = check_contains(task, conf_tasks)
                    if index != -1:
                        self.state[index][diff] = 2
            diff += 1

        diff = 0
        array_index = len(conf_tasks)
        # Append tasks not in conflict
        for d in range(left_bound, right_bound):
            day = self.calendar[d]
            if conf_tasks[0].type == "c-check":
                for t in day.c_checks:
                    if check_contains(t, conf_tasks) == -1:
                        self.state[array_index][diff] += 1

            elif conf_tasks[0].type == "a-check":
                for t in day.a_checks:
                    if check_contains(t, conf_tasks) == -1:
                        self.state[array_index][diff] += 1
            diff += 1

    # adds a task to the respective aircraft list (replaces the task if it already exists)
    def add_task_to_aircraft(self, task):
        for index in range(len(self.aircraft)):
            if task.tail_number == self.aircraft[index].tail_number:
                if task.type == "a-check":
                    # if task exists replace it
                    for i in range(len(self.aircraft[index].a_checks)):
                        if task.id == self.aircraft[index].a_checks[i].id:
                            self.aircraft[index].a_checks[i] = task
                            return
                    self.aircraft[index].a_checks.append(task)
                    return
                else:
                    # if task exists replace it
                    for i in range(len(self.aircraft[index].c_checks)):
                        if task.id == self.aircraft[index].c_checks[i].id:
                            self.aircraft[index].c_checks[i] = task
                            return
                    self.aircraft[index].c_checks.append(task)
                    return

    # configure initial tasks with due dates, starting date, end date and previous check
    def setup_initial_task(self, task, first):
        # get interval from respective aircraft
        interval = get_task_interval(task, self.aircraft)
        # if the task is the first one to be scheduled for that airplane we have to consider the current usage ratio
        if first:
            start_date = date_to_day(START_DATE)
            dy_int = interval.get("DY") - task.initial_check_usage.get("DY")
            fh_int = interval.get("FH") - task.initial_check_usage.get("FH")
            fc_int = interval.get("FC") - task.initial_check_usage.get("FC")
            new_int = {"DY": dy_int, "FH": fh_int, "FC": fc_int}

            task = self.calculate_due_date(task, start_date, new_int)
        else:
            # prev_check is the end day of the last task done on the aircraft
            task.prev_check = date_to_day(get_task_prev_check(task, self.aircraft))
            start_date = task.prev_check

            task = self.calculate_due_date(task, start_date, interval)

        task.starting_day = task.due_date
        task.end_day = task.starting_day + timedelta(task.length - 1)
        self.calculate_task_losses(task)
        return task

    # calculates next due date based on DY, FH and FC (whatever comes first)
    def calculate_due_date(self, task, prev_check, interval):
        dfh = None
        dfc = None
        for a in self.aircraft:
            if task.tail_number == a.tail_number:
                dfh = a.dfh
                dfc = a.dfc
                break

        # calculate due date based on calendar days
        task.dy_due_date = day_to_date(prev_check + interval.get("DY"))

        # calculate due date based on flight hours
        fh_usage = 0
        fh_date = day_to_date(prev_check)
        while fh_usage < interval.get("FH"):
            fh_date += timedelta(1)
            fh_usage += dfh[fh_date.month - 1]
        task.fh_due_date = fh_date - timedelta(1)

        # calculate due date based on flight cycles
        fc_usage = 0
        fc_date = day_to_date(prev_check)
        while fc_usage < interval.get("FC"):
            fc_date += timedelta(1)
            fc_usage += dfc[fc_date.month - 1]
        task.fc_due_date = fc_date - timedelta(1)
        task.due_date = day_to_date(min([date_to_day(task.fc_due_date), date_to_day(task.fh_due_date),
                                         date_to_day(task.dy_due_date)]))
        return task

    # calculates dy_lost, fh_lost and fc_lost for a specific task
    def calculate_task_losses(self, task):
        if task.starting_day == task.due_date:
            task.dy_lost = 0
            task.fh_lost = 0
            task.fc_lost = 0
        else:
            task.dy_lost = date_to_day(task.due_date) - date_to_day(task.starting_day)
            sum_fh_lost = 0
            sum_fc_lost = 0

            for a in self.aircraft:
                if a.tail_number == task.tail_number:
                    if task.starting_day < task.due_date:
                        for day in range(date_to_day(task.starting_day), date_to_day(task.due_date)):
                            sum_fh_lost += a.dfh[self.calendar[day].date.month - 1]
                            sum_fc_lost += a.dfc[self.calendar[day].date.month - 1]
                    else:
                        for day in range(date_to_day(task.due_date), date_to_day(task.starting_day)):
                            sum_fh_lost -= a.dfh[self.calendar[day].date.month - 1]
                            sum_fc_lost -= a.dfc[self.calendar[day].date.month - 1]
                    task.fh_lost = int(sum_fh_lost)
                    task.fc_lost = int(sum_fc_lost)
                    break

    def render(self, mode='human'):
        pass


# checks if the check can be done at a certain day
def check_day_availability(day, task):
    if day.c_check_available == False and day.a_check_available == False:
        return False
    if day.a_check_available == False and task.type == "a-check":
        return False
    if day.c_check_available == False and task.type == "c-check":
        return False
    return True


def calculate_rewards(tasks):
    reward = 0

    for task in tasks:
        start_day = date_to_day(task.starting_day)
        due_date = date_to_day(task.due_date)

        if start_day == -1:
            reward -= 100
        elif start_day <= due_date:
            reward -= task.fh_lost * math.pow(MAX_PRIORITY - task.priority + 1, 2)
        else:
            reward -= -10 * task.fh_lost * math.pow(MAX_PRIORITY - task.priority + 1, 2)
    return reward


def check_contains(task, array):
    for i in range(len(array)):
        t = array[i]
        if t.id == task.id:
            return i
    return -1


# check if task exists on the aircraft
def check_aircraft(aircraft, task):
    for a in aircraft:
        if task.tail_number == a.tail_number:
            if task.type == "a-check":
                for t in a.a_checks:
                    if task.id == t.id:
                        return True
                return False
            else:
                for t in a.c_checks:
                    if task.id == t.id:
                        return True
                return False


# retrieves the task interval for a specific aircraft
def get_task_interval(task, aircraft):
    for a in aircraft:
        if task.tail_number == a.tail_number:
            if task.type == "a-check":
                return a.a_check_interval
            else:
                return a.c_check_interval


# retrieves the previous check of a specific aircraft
def get_task_prev_check(task, aircraft):
    for a in aircraft:
        if task.tail_number == a.tail_number:
            if task.type == "a-check":
                return a.a_checks[-1].end_day
            else:
                return a.c_checks[-1].end_day


# verify if all scheduled tasks have proper spacing
def verify_task_due_dates(tasks, aircraft):
    for i in range(len(tasks)):
        for j in range(i+1, len(tasks)):
            if tasks[i].tail_number == tasks[j].tail_number and tasks[i].type == tasks[j].type:
                if date_to_day(tasks[j].due_date) != date_to_day(tasks[i].end_day) + get_task_interval(tasks[i], aircraft):
                    print("Error:")
                    print("Task id: ", tasks[i].id,", end day: ", date_to_day(tasks[i].end_day),
                          ", interval: ", get_task_interval(tasks[i], aircraft))
                    print("Task id: ", tasks[j].id,", due date: ", date_to_day(tasks[j].due_date))
                break


def calculate_aircraft_loss(aircraft_list, task_list):
    for aircraft in aircraft_list:
        sum_lost_dy = 0
        sum_lost_fh = 0
        sum_lost_fc = 0
        for check in task_list:
            if check.tail_number == aircraft.tail_number:
                sum_lost_dy += check.dy_lost
                sum_lost_fh += check.fh_lost
                sum_lost_fc += check.fc_lost
        aircraft.dy_lost = sum_lost_dy
        aircraft.fh_lost = sum_lost_fh
        aircraft.fc_lost = sum_lost_fc


if __name__ == '__main__':
    e = SchedulingEnv()

