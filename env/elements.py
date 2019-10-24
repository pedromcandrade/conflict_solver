import numpy as np


class Day:
    def __init__(self, date):
        self.date = date

        self.c_checks = []
        self.a_checks = []

        self.a_check_available = True
        self.c_check_available = True


class Hangar:
    def __init__(self, n_days, n_tasks):
        self.n_tasks = n_tasks
        self.n_days = n_days
        self.calendar = np.zeros((n_tasks, n_days))

    def available_day(self, task, due_date, length):
        for i in range(due_date, length-2, -1):
            if all(v == 0 for v in self.calendar[task][i-length+1:i+1]):
                return i
        return -1

    def available_day_after(self, due_date, length):
        d = -1
        for i in range(due_date-length+1, self.n_days):
            possible = True
            for t in range(self.n_tasks):
                if all(v == 0 for v in self.calendar[t][i:i+length]):
                    d = i
                else:
                    possible = False

            if possible:
                return d + length - 1
        return -1

    def schedule_maintenance(self, task, day, length):
        self.calendar[task][day-length+1:day+1] = 1


class Task:
    def __init__(self, id, type, length, interval, tolerance, prev_check, number, fleet, tail_number, priority):
        self.id = id
        self.type = type
        self.length = length
        self.interval = interval
        self.tolerance = tolerance
        self.number = number
        self.prev_check = prev_check

        self.fleet = fleet
        self.tail_number = tail_number

        self.due_date = -1
        self.starting_day = -1
        self.end_day = -1
        self.hangar = -1

        self.scheduled = False
        self.priority = priority

        self.up = False


class Aircraft:
    def __init__(self, fleet, tail_number):
        self.fleet = fleet
        self.tail_number = tail_number
        self.c_checks = []
        self.a_checks = []


class Conflict:
    def __init__(self, tasks):
        self.tasks = tasks
