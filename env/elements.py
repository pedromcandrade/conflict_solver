
from env.agent import *
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
    def __init__(self, id, type, length, tolerance, dy_prev_check, fh_prev_check, fc_prev_check, number, fleet, tail_number, priority):
        self.id = id
        self.type = type
        self.length = length
        self.tolerance = tolerance
        self.number = number
        self.initial_check_usage = {"DY": dy_prev_check, "FH": fh_prev_check, "FC": fc_prev_check}
        self.prev_check = -1

        self.fleet = fleet
        self.tail_number = tail_number

        self.due_date = -1
        self.starting_day = -1
        self.end_day = -1
        self.hangar = -1

        self.dy_due_date = -1
        self.fh_due_date = -1
        self.fc_due_date = -1

        self.scheduled = False
        self.priority = priority

        self.dy_lost = 0
        self.fh_lost = 0
        self.fc_lost = 0

        self.up = False


class Aircraft:
    def __init__(self, fleet, tail_number, a_check_interval_dy, a_check_interval_fh, a_check_interval_fc,
                 c_check_interval_dy, c_check_interval_fh, c_check_interval_fc, dfh, dfc):
        self.fleet = fleet
        self.tail_number = tail_number
        self.c_checks = []
        self.a_checks = []
        self.a_check_interval = {"FH": a_check_interval_fh, "FC": a_check_interval_fc, "DY": a_check_interval_dy}
        self.c_check_interval = {"FH": c_check_interval_fh, "FC": c_check_interval_fc, "DY": c_check_interval_dy}
        self.dfh = dfh
        self.dfc = dfc
        self.dy_lost = 0
        self.fh_lost = 0
        self.fc_lost = 0


class Conflict:
    def __init__(self, tasks):
        self.tasks = tasks
