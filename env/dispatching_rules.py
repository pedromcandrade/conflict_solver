# Earliest Due Date: EDD (changed to Earliest Scheduled Day)
# Tardiest Due Date: TDD (changed to Tardiest Scheduled Day)
# Shortest Length First: SLF
# Longest Length First: LLF

from env.utils import *

list_rules = ["EDD", "FCFS", "MUF"]


def n_rules():
    return len(list_rules)


def edd(tasks):
    bigger = 9999
    m = None

    for t in tasks:
        if date_to_day(t.due_date) < bigger:
            bigger = date_to_day(t.due_date)
            m = t
    return m


def tdd(tasks):
    bigger = -9999
    m = None
    for t in tasks:
        if date_to_day(t.due_date) > bigger:
            bigger = date_to_day(t.due_date)
            m = t
    return m


def slf(tasks):
    bigger = 9999
    m = None
    for t in tasks:
        if t.length < bigger:
            bigger = t.length
            m = t
    return m


def llf(tasks):
    bigger = -9999
    m = None
    for t in tasks:
        if t.length > bigger:
            bigger = t.length
            m = t
    return m
