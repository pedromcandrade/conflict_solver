import gym
import copy
from collections import deque
from gym import spaces
import random
from env.utils import *
from env.elements import *
from env.dispatching_rules import *
import math

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
        self.n_days = 2190
        self.n_tasks = len(self.c_initial) + len(self.a_initial)
        self.a_slots = 1
        self.c_slots = 3
        self.a_length = 1
        self.c_length = 21
        self.c_interval = 3

        self.c_tasks_before = 0
        self.c_tasks_after = 0
        self.a_tasks_before = 0
        self.a_tasks_after = 0
        self.conf_number = 0
        self.c_check_conf_number = 0
        self.a_check_conf_number = 0

        self.last_c_days = [task[4] for task in self.c_initial]
        self.last_elapsed_index = np.zeros(len(self.c_initial))
        self.last_a_days = [task[4] for task in self.a_initial]
        # Calendar
        today_date = date.today()
        self.calendar = []
        for i in range(self.n_days):
            self.calendar.append(Day(today_date + timedelta(days=i)))

        self.aircraft = []
        self.task_list = []
        self.conflicts = deque()
        self.last_scheduled = 0
        self.done = False

        self.observation_space = spaces.Box(low=0, high=1, shape=(15, 100))  # Partial Calendar
        self.state = None

        self.action_space = spaces.Discrete(4)  # Number of dispatching rules

        self.more_tasks = False

        self.init_aircraft()
        self.create_tasks_v2()
        #random.shuffle(self.task_list)
        build_input_json(self.task_list)

        self.backup = copy.deepcopy(self.task_list)

    def reward_calendar(self):
        reward = 0

        for task in self.task_list:
            starting_day = date_to_day(task.starting_day)
            due_day = date_to_day(task.due_date)
            if starting_day == -1:
                reward -= 100
            elif starting_day <= due_day:
                reward -= (due_day - starting_day) * math.pow(MAX_PRIORITY - task.priority + 1, 2)

                if task.type == "c-check":
                    self.c_tasks_before += 1
                else:
                    self.a_tasks_before += 1

            else:
                reward -= 10 * (starting_day - due_day) * math.pow(MAX_PRIORITY - task.priority + 1, 2)
                if task.type == "c-check":
                    self.c_tasks_after += 1
                else:
                    self.a_tasks_after += 1

        return reward

    def reset(self):
        self.c_tasks_before = 0
        self.c_tasks_after = 0

        self.a_tasks_before = 0
        self.a_tasks_after = 0

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
        # self.init_aircraft()

        self.n_tasks = len(self.c_initial) + len(self.a_initial)
        self.a_slots = 1
        self.c_slots = 3
        self.a_length = 1
        self.c_length = 21
        self.c_interval = 3

        self.last_c_days = [task[4] for task in self.c_initial]
        self.last_a_days = [task[4] for task in self.a_initial]

        # Calendar
        today_date = date.today()
        self.calendar = []
        for i in range(self.n_days):
            self.calendar.append(Day(today_date + timedelta(days=i)))

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

        self.move_maintenance(move)

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

        reward = self.calculate_rewards(conflict.tasks)

        return self.state, reward, self.done, {}

    def schedule_maintenance(self, task, move=False):
        self.more_tasks = True

        if move:
            diff = date_to_day(task.starting_day)
        else:
            diff = date_to_day(task.due_date)
            task.starting_day = task.due_date
            task.end_day = task.starting_day + timedelta(task.length - 1)
            task.scheduled = True

        t_type = task.type

        if t_type == "a-check":
            self.calendar[diff].a_checks.append(task)

        elif t_type == "c-check":  # Schedule c-check in the several days
            for i in range(date_to_day(task.starting_day), date_to_day(task.end_day) + 1):
                self.calendar[i].c_checks.append(task)

        # Checking conflicts
        self.check_conflicts(task)

        # Change due date following task
        for i in range(task.number + 1, len(self.task_list), 1):
            # use the id to verify the next tasks
            if self.task_list[i].id[:-1] == task.id[:-1]:
                self.task_list[i].due_date = task.end_day + timedelta(days=task.interval)

                if task.starting_day < task.due_date and self.task_list[i].scheduled:
                    self.delete_conflicts(self.task_list[i])
                    self.delete_from_calendar(self.task_list[i])

                    self.task_list[i].scheduled = False
                else:
                    break

    def move_maintenance(self, task):
        self.delete_conflicts(task)  # delete conflicts with this task

        # Find new day
        found = True
        previous_start_day = date_to_day(task.starting_day)
        previous_end_day = date_to_day(task.end_day)

        if task.type == "a-check":
            new_day = task.starting_day - timedelta(1)

            if date_to_day(new_day) >= 0 and not task.up:
                task.starting_day = new_day
                task.end_day = task.starting_day + timedelta(task.length - 1)

            else:
                # enters here only if the task could not be scheduled before its due date
                task.up = True
                new_day = task.starting_day + timedelta(1)
                if new_day < task.due_date + timedelta(task.tolerance) and new_day + task.length < day_to_date(self.n_days):
                    task.starting_day = new_day
                    task.end_day = task.starting_day + timedelta(task.length - 1)
                else:
                    print("not found:", task.number, date_to_day(task.due_date))
                    found = False

        elif task.type == "c-check":
            new_day = self.find_c_day(task)
            if date_to_day(new_day) >= 0 and not task.up:
                task.starting_day = new_day
                task.end_day = task.starting_day + timedelta(task.length - 1)
            else:
                # enters here only if the task could not be scheduled before its due date
                task.up = True
                new_day = self.find_c_tolerance(task)
                if new_day < task.due_date + timedelta(task.tolerance) and new_day + task.length < day_to_date(self.n_days):
                    task.starting_day = new_day
                    task.end_day = task.starting_day + timedelta(task.length - 1)
                else:
                    print("not found:", task.number, date_to_day(task.due_date))
                    found = False

        #self.delete_from_calendar(task)

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

        if found:
            self.schedule_maintenance(task, move=True)
        else:
            task.starting_day = day_to_date(-1)
            task.end_day = day_to_date(-1)

    # verify if all tasks are scheduled (returns the index of the first task not scheduled and -1 if all are scheduled)
    def verify_tasks_scheduled(self):
        for index in range(len(self.task_list)):
            if not self.task_list[index].scheduled:
                return index
        return -1

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
                        if self.check_containts(t, new_conf.tasks) == -1:
                            new_conf.tasks.append(t)
            if len(new_conf.tasks) > 3:
                self.conflicts.append(new_conf)

            elif task.type == "a-check":
                if len(self.calendar[start_day].a_checks) > 1:
                    new_conf = Conflict(self.calendar[start_day].a_checks)
                    self.conflicts.append(new_conf)

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

    def simulate_move(self, task, task_list):
        new_day = None
        # Find new day
        if task.type == "a-check":
            new_day = task.starting_day - timedelta(1)

            if date_to_day(new_day) >= self.a_length and not task.up:
                new_day = task.starting_day - timedelta(1)

            else:
                new_day = task.starting_day + timedelta(1)
                if new_day < task.due_date + timedelta(task.tolerance) and new_day + task.length < day_to_date(self.n_days):
                    new_day = task.starting_day + timedelta(1)
                else:
                    new_day = date.today() - timedelta(50)

        elif task.type == "c-check":
            new_day = self.find_c_day(task)

            if date_to_day(new_day) >= task.length and not task.up:
                new_day = self.find_c_day(task)

            else:
                new_day = self.find_c_tolerance(task)
                if new_day < task.due_date + timedelta(task.tolerance) and new_day + task.length < day_to_date(self.n_days):
                    new_day = self.find_c_tolerance(task)
                else:
                    new_day = date.today() - timedelta(50)

        reward = 0
        for t in task_list:
            if t.id != task.id:
                day = date_to_day(t.starting_day)
            else:
                day = date_to_day(new_day)

            due_date = date_to_day(t.due_date)

            if day <= due_date:
                reward -= due_date - day
            else:
                reward -= 5 * (day - due_date)

        return reward

    def init_aircraft(self):
        self.aircraft = []
        for a in range(len(self.c_initial)):
            self.aircraft.append(Aircraft(self.c_initial[a][0], self.c_initial[a][1]))

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
                new_task = Task(str(task_info[1]) + "c" + str(c_check_index - 2), "c-check", t_length, task_info[7],
                                task_info[10], task_info[4], task_number, task_info[0], task_info[1], task_priority)
                if first:
                    new_task.prev_check = - new_task.prev_check
                    new_task.due_date = (date.today() + timedelta(new_task.interval + new_task.prev_check))
                    new_task.starting_day = new_task.due_date
                    new_task.end_day = new_task.starting_day + timedelta(new_task.length - 1)
                    first = False
                else:
                    # prev_check is the end day of the last task done on the aircraft
                    new_task.prev_check = date_to_day(self.aircraft[index].c_checks[-1].end_day)
                    new_task.due_date = day_to_date(new_task.prev_check + new_task.interval)
                    new_task.starting_day = new_task.due_date
                    new_task.end_day = new_task.starting_day + timedelta(new_task.length - 1)

                # If maintenance already passed its due date (rare) (case when t = 47)
                if date_to_day(new_task.due_date) < 0:
                    new_task.due_date = day_to_date(t_length)
                    new_task.starting_day = new_task.due_date
                    new_task.end_day = new_task.starting_day + timedelta(new_task.length - 1)

                # If due date is less than the number of days the task is valid
                if date_to_day(new_task.end_day) < self.n_days:
                    self.task_list.append(new_task)  # add new task to the global task list
                    if not self.check_aircraft(self.aircraft[index], new_task):
                        self.aircraft[index].c_checks.append(new_task)  # add new task to the respective aircraft task list
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
                new_task = Task(str(task_info[1]) + "a" + str(count), "a-check", self.a_length, task_info[7],
                                task_info[10], task_info[4], task_number, task_info[0], task_info[1], task_priority)
                if first:
                    new_task.prev_check = - new_task.prev_check
                    new_task.due_date = (date.today() + timedelta(new_task.interval + new_task.prev_check))
                    new_task.starting_day = new_task.due_date
                    new_task.end_day = new_task.starting_day + timedelta(new_task.length - 1)
                    first = False
                else:
                    # prev_check is the end day of the last task done on the aircraft
                    new_task.prev_check = date_to_day(self.aircraft[index].a_checks[-1].end_day)
                    new_task.due_date = day_to_date(new_task.prev_check + new_task.interval)
                    new_task.starting_day = new_task.due_date
                    new_task.end_day = new_task.starting_day + timedelta(new_task.length - 1)

                # If maintenance already passed its due date (rare)
                if date_to_day(new_task.due_date) < 0:
                    new_task.due_date = day_to_date(t_length)
                    new_task.starting_day = new_task.due_date
                    new_task.end_day = new_task.starting_day + timedelta(new_task.length - 1)

                # If due date is less than the number of days
                if date_to_day(new_task.end_day) < self.n_days:
                    self.task_list.append(new_task)  # add new task to the global task list
                    if not self.check_aircraft(self.aircraft[index], new_task):
                        self.aircraft[index].a_checks.append(new_task)  # add new task to the respective aircraft task list

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
                    new_task = Task(str(task_info[1]) + "c" + str(index), "c-check", t_length, task_info[7],
                                    task_info[10], task_info[4], task_number, task_info[0], task_info[1], task_priority)

                    if index == 0:
                        new_task.prev_check = - new_task.prev_check
                        new_task.due_date = (date.today() + timedelta(new_task.interval + new_task.prev_check))
                        new_task.starting_day = new_task.due_date
                        new_task.end_day = new_task.starting_day + timedelta(new_task.length - 1)
                    else:
                        # prev_check is the end day of the last task done on the aircraft
                        new_task.prev_check = date_to_day(self.aircraft[t].c_checks[-1].end_day)
                        new_task.due_date = day_to_date(new_task.prev_check + new_task.interval)
                        new_task.starting_day = new_task.due_date
                        new_task.end_day = new_task.starting_day + timedelta(new_task.length - 1)

                    # If maintenance already passed its due date its gonna be less than zero (rare)
                    if date_to_day(new_task.due_date) < 0:
                        new_task.due_date = day_to_date(t_length)
                        new_task.starting_day = new_task.due_date
                        new_task.end_day = new_task.starting_day + timedelta(new_task.length - 1)

                    # If due date is less than the number of days
                    if date_to_day(new_task.end_day) < self.n_days:
                        self.task_list.append(new_task)
                        if not self.check_aircraft(self.aircraft[t], new_task):
                            self.aircraft[t].c_checks.append(new_task)  # add new task to the aircraft task list
                        c_checks += 1
                        more_tasks = True
                        task_number += 1

            # A-checks
            for t in range(len(self.a_initial)):
                task_info = self.a_initial[t]
                t_length = self.a_length
                # define random priority
                task_priority = random.randint(1, MAX_PRIORITY)
                new_task = Task(str(task_info[1]) + "a" + str(index), "a-check", t_length, task_info[7],
                                task_info[10], task_info[4], task_number, task_info[0], task_info[1],
                                task_priority)

                if index == 0:
                    new_task.prev_check = - new_task.prev_check
                    new_task.due_date = (date.today() + timedelta(new_task.interval + new_task.prev_check))
                    new_task.starting_day = new_task.due_date
                    new_task.end_day = new_task.starting_day + timedelta(new_task.length - 1)
                else:
                    # prev_check is the end day of the last task done on the aircraft
                    new_task.prev_check = date_to_day(self.aircraft[t].a_checks[-1].end_day)
                    new_task.due_date = day_to_date(new_task.prev_check + new_task.interval)
                    new_task.starting_day = new_task.due_date
                    new_task.end_day = new_task.starting_day + timedelta(new_task.length - 1)

                # If maintenance already passed its due date its gonna be less than zero (rare) (case when t = 47)
                if date_to_day(new_task.due_date) < 0:
                    new_task.due_date = day_to_date(t_length)
                    new_task.starting_day = new_task.due_date
                    new_task.end_day = new_task.starting_day + timedelta(new_task.length - 1)

                # If due date is less than the number of days
                if date_to_day(new_task.end_day) < self.n_days:
                    self.task_list.append(new_task)
                    if not self.check_aircraft(self.aircraft[t], new_task):
                        self.aircraft[t].a_checks.append(new_task)  # add new task to the respective aircraft task list
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
                    index = self.check_containts(task, conf_tasks)
                    if index != -1:
                        self.state[index][diff] = 2
            elif conf_tasks[0].type == "a-check":
                checks = sorted(day.a_checks, key=lambda x: x.number)
                for t in range(len(checks)):
                    task = checks[t]
                    index = self.check_containts(task, conf_tasks)
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
                    if self.check_containts(t, conf_tasks) == -1:
                        self.state[array_index][diff] += 1

            elif conf_tasks[0].type == "a-check":
                for t in day.a_checks:
                    if self.check_containts(t, conf_tasks) == -1:
                        self.state[array_index][diff] += 1
            diff += 1
        """
        array_index = len(conf_tasks)
        for i in range(len(not_conflicts)):
            t = not_conflicts[i]

            t_index = date_to_day(t.starting_day) - left_bound

            for k in range(t_index, t_index + t.length):
                if k < 100 and k > 0:
                    self.state[array_index][k] += 1
        """
        # np.set_printoptions(threshold=sys.maxsize)
        # print(self.state)

    def check_containts(self, task, array):
        for i in range(len(array)):
            t = array[i]
            if t.id == task.id:
                return i
        return -1

    def check_aircraft(self, aircraft, task):
        if task.type == "a-check":
            for t in aircraft.a_checks:
                if task.id == t.id:
                    return True
            return False
        else:
            for t in aircraft.c_checks:
                if task.id == t.id:
                    return True
            return False

    def calculate_rewards(self, tasks):
        reward = 0

        for task in tasks:
            start_day = date_to_day(task.starting_day)
            due_date = date_to_day(task.due_date)

            if start_day == -1:
                reward -= 100
            elif start_day <= due_date:
                reward -= (due_date - start_day) * math.pow(MAX_PRIORITY - task.priority + 1, 2)
            else:
                reward -= 10 * (start_day - due_date) * math.pow(MAX_PRIORITY - task.priority + 1, 2)
        return reward

    def render(self, mode='human'):
        pass


if __name__ == '__main__':
    e = SchedulingEnv()
    e.reset()
    e.create_tasks()
    tasks = e.task_list
    e.reset()
    e.create_tasks_v2()
    tasks2 = e.task_list

    for t in tasks:
        for t2 in tasks2:
            if t.id == t2.id:
                if t.due_date != t2.due_date or t.starting_day != t2.starting_day or t.end_day != t2.end_day:
                    print(t.id)
                    print(t.due_date, "--", t2.due_date)
                    print(t.starting_day, "--", t2.starting_day)
                    print(t.end_day, "--", t2.end_day)


