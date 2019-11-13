from env.agent import *
import simplejson as json
import xlsxwriter
import pandas as pd
import numpy as np
from datetime import *


START_DATE = date(2019, 1, 1)


def date_to_day(d):
    day = (d - START_DATE).days
    return day


def day_to_date(day):
    return START_DATE + timedelta(day)


def xlxs_to_matrix(filename):
    df = pd.read_excel(filename)
    matrix = df.as_matrix()
    matrix = np.asarray(matrix)
    for l in matrix:
        l[0] -= 1
        l[2] -= 1

    return matrix


def file_to_info(filename):
    names = ["D_INITIAL", "C_INITIAL", "A_INITIAL", "DFH", "DFC", "C_ELAPSED_TIME",
             "ADDITIONAL", "C_NOT_ALLOWED", "MORE_C_SLOTS", "PUBLIC_HOLIDAYS",
             "A_NOT_ALLOWED", "MORE_A_SLOTS"]
    info = []

    for i in range(12):
        df = pd.read_excel(filename, sheet_name=names[i])
        matrix = np.asarray(df.to_numpy())
        info.append(matrix)

    return info


def render_calendar_excel(env, name):
    workbook = xlsxwriter.Workbook(name)
    ws = workbook.add_worksheet()
    info = workbook.add_worksheet()

    date_format = workbook.add_format()
    date_format.set_pattern(1)
    date_format.set_bg_color('#333333') # dark gray
    date_format.set_font_color('white')
    date_format.set_bold()

    c_format = workbook.add_format()
    c_format.set_pattern(1)
    c_format.set_bg_color('#00a000') #green

    c_format_before_dd = workbook.add_format()
    c_format_before_dd.set_pattern(1)
    c_format_before_dd.set_bg_color("#00ff00") #light green

    a_format = workbook.add_format()
    a_format.set_pattern(1)
    a_format.set_bg_color('#1e90ff') #blue

    a_format_before_dd = workbook.add_format()
    a_format_before_dd.set_pattern(1)
    a_format_before_dd.set_bg_color("#87cfeb") #light blue

    tolerance_used_format = workbook.add_format()
    tolerance_used_format.set_pattern(1)
    tolerance_used_format.set_bg_color('red')

    no_check_format = workbook.add_format()
    no_check_format.set_pattern(1)
    no_check_format.set_bg_color("#c0c0c0") #light gray

    row = 0
    column = 0
    for d in range(len(env.calendar)):
        day = env.calendar[d]

        ws.write(row, column, str(day.date), date_format)  # write date

        if not day.c_check_available:
            ws.write(row + 1, column, '', no_check_format)
            ws.write(row + 2, column, '', no_check_format)
            ws.write(row + 3, column, '', no_check_format)

        # ws.write(row, column, date_to_day(day.date))
        row += 1

        sorted_list = sorted(day.c_checks, key=lambda x: x.number)

        for c in range(len(sorted_list)):  # paint c_checks
            if not day.c_check_available:
                ws.write(row + c, column, sorted_list[c].number, no_check_format)
            else:
                if sorted_list[c].starting_day == sorted_list[c].due_date:
                    ws.write(row + c, column, sorted_list[c].number, c_format)
                elif sorted_list[c].starting_day < sorted_list[c].due_date:
                    ws.write(row + c, column, sorted_list[c].number, c_format_before_dd)
                else:
                    ws.write(row + c, column, sorted_list[c].number, tolerance_used_format)

        row += 3
        if not day.a_check_available:
            ws.write(row, column, '', no_check_format)
        for c in range(len(day.a_checks)):  # paint a_checks
            if not day.a_check_available:
                ws.write(row, column, day.a_checks[c].number, no_check_format)
            else:
                if day.a_checks[c].starting_day == day.a_checks[c].due_date:
                    ws.write(row, column, day.a_checks[c].number, a_format)
                elif day.a_checks[c].starting_day < day.a_checks[c].due_date:
                    ws.write(row, column, day.a_checks[c].number, a_format_before_dd)
                else:
                    ws.write(row, column, day.a_checks[c].number, tolerance_used_format)

        if d % 50 != 0 or d == 0:  # don't change line
            column += 1
            row -= 4
        else:  # change line
            row += 1
            column = 0

    ws.set_column('A:AZ', 10)

    row = 0
    column = 0
    for t in env.task_list:
        info.write(row, column, t.number)
        if t.starting_day != -1:
            diff = (t.due_date - t.starting_day).days
            if diff < 0:
                info.write(row, column + 1, diff, tolerance_used_format)
            elif t.type == "c-check" and diff > 0:
                info.write(row, column + 1, diff, c_format_before_dd)
            elif t.type == "a-check" and diff > 0:
                info.write(row, column + 1, diff, a_format_before_dd)
            elif t.type == "c-check" and diff == 0:
                info.write(row, column + 1, diff, c_format)
            else:
                info.write(row, column + 1, diff, a_format)
        else:
            info.write(row, column + 1, "Not Found")

        # info.write(row, column + 2, t.type)

        if t.number % 30 != 0 or t.number == 0:  # dont change line
            column += 2
        else:
            row += 1
            column = 0

    workbook.close()


def render_results_excel(agent, mean_rewards, max_reward, episode_conflicts, best_episode_conflicts, c_before,
                         c_after, a_before, a_after):
    workbook = xlsxwriter.Workbook('results.xlsx')
    worksheet = workbook.add_worksheet()
    # hyperparameters = workbook.add_worksheet()
    worksheet.write_row(0, 0, mean_rewards)
    hyper_strings = ["n.layers", "layer1", "layer2", "layer3", "activation", "initializer",
                     "output activation", "optimizer", "gamma", "buffer_size", "batch_size", "target_update",
                     "learning rate", "max frames"]
    hyper_values = [n_layers, layer1nodes, layer2nodes, layer3nodes, activation_function,
                    "tf.contrib.layers.variance_scaling_initializer()",
                    "None", "adam", agent.gamma, agent.buffer_size, agent.batch_size, agent.target_update_freq,
                    agent.learning_rate, agent.max_frames]
    worksheet.write_row(2, 0, hyper_strings)
    worksheet.write_row(3, 0, hyper_values)

    max_value = ["max value", max_reward]
    worksheet.write_row(5, 0, max_value)

    conflicts = ["Mean conflicts:", sum(episode_conflicts) / len(episode_conflicts),
                 "Number of conflicts in best episode:", best_episode_conflicts]
    worksheet.write_row(7, 0, conflicts)

    tasks_position = ["C tasks before:", c_before, "C tasks after:", c_after, "A tasks before:", a_before,
                      "A tasks after:", a_after]
    worksheet.write_row(8, 0, tasks_position)

    workbook.close()


def create_json(env):
    d = {}
    # go through all aircraft
    for i in range(env.c_initial.shape[0]):
        aircraft = env.c_initial[i][1]
        # find all tasks related with this aircraft
        task_list = []
        for task in env.task_list:
            if aircraft == task.tail_number:
                task_info = {"id": task.id, "number": task.number, "type": task.type, "priority": task.priority,
                             "due date": str(task.due_date), "length": task.length,
                             "starting day": str(task.starting_day), "end day": str(task.end_day),
                             "aircraft": task.tail_number, "fleet": task.fleet,
                             "DY_LOST": task.dy_lost, "FH_LOST": task.fh_lost, "FC_LOST": task.fc_lost}
                task_list.append(task_info)
        d[aircraft] = task_list

    j = json.dumps(d, indent=4)
    with open('data.json', 'w') as f:
        f.write(j)


def build_input_json(tasks):
    d = {}
    for task in tasks:
        d[task.id] = {"due date": str(task.due_date), "tolerance": str(task.tolerance), "priority": str(task.priority)}
    j = json.dumps(d, indent=4)
    with open('input.json', 'w') as f:
        f.write(j)


def task_oriented_json(env):
    d = {}
    sorted_tasks = sorted(env.task_list, key=lambda x: date_to_day(x.due_date))
    for task in sorted_tasks:
        task_info = {"number": task.number, "type": task.type, "due date": str(task.due_date), "priority": task.priority,
                     "length": task.length, "starting day": str(task.starting_day), "end day": str(task.end_day),
                     "aircraft": task.tail_number, "fleet": task.fleet, "DY_LOST": task.dy_lost, "FH_LOST": task.fh_lost,
                     "FC_LOST": task.fc_lost}
        d[task.id] = task_info

    j = json.dumps(d, indent=4)
    with open('tasks.json', 'w') as f:
        f.write(j)


def maintenance_plan_json(env):
    d = {}
    calendar = env.calendar
    for day in calendar:
        # c_check_ids = [task.id for task in day.c_checks]
        # a_check_ids = [task.id for task in day.a_checks]
        c_task = {}
        a_task = {}
        for task in day.c_checks:
            if day.date == task.end_day:
                c_task[task.id] = {"state": "1"}
            else:
                c_task[task.id] = {"state": "0"}
        for task in day.a_checks:
            if day.date == task.end_day:
                a_task[task.id] = {"state": "1"}
            else:
                a_task[task.id] = {"state": "0"}
        day_info = {"c-checks": c_task, "a-checks": a_task}
        d[str(day.date)] = day_info

    j = json.dumps(d, indent=4)
    with open('plan.json', 'w') as f:
        f.write(j)


# finds if an a-check could be scheduled in a later slot before its due date (considering all tasks have equal priority)
def evaluate_calendar(tasks, calendar):
    count = 0
    for task in tasks:
        sub_optimal = False
        if task.starting_day < task.due_date:
            for i in range(date_to_day(task.starting_day), date_to_day(task.due_date) + 1):
                if task.type == "a-check":
                    if calendar[i].a_check_available and len(calendar[i].a_checks) < 1:
                        sub_optimal = True
                        print("Task ", task.number, "/", task.id, " not optimal. Available day: ", calendar[i].date)
        if sub_optimal:
            count += 1
    print("Total number of sub-optimal tasks: ", count)


# finds if the tasks with the same due date were scheduled properly, i.e, if the tasks with
# higher priority were scheduled first
def evaluate_priorities():
    with open("tasks.json") as json_file:
        tasks = json.load(json_file)
        current_due_date = 0
        wrong_tasks = []
        current_tasks = []
        for task_id in tasks:
            temp_due_date = date_to_day(datetime.strptime(tasks[task_id].get("due date"), '%Y-%m-%d').date())
            if current_due_date == temp_due_date:
                current_tasks.append([task_id, tasks[task_id].get("priority"), tasks[task_id].get("starting day")])
            else:
                if len(current_tasks) > 1:
                    for i in range(len(current_tasks)):
                        for j in range(i, len(current_tasks)):
                            if current_tasks[i][1] < current_tasks[j][1] and current_tasks[i][2] < current_tasks[j][2]:
                                if current_tasks[i] not in wrong_tasks:
                                    wrong_tasks.append(current_tasks[i])
                                if current_tasks[j] not in wrong_tasks:
                                    wrong_tasks.append(current_tasks[j])
                current_due_date = temp_due_date
                current_tasks = []

    print(wrong_tasks)


# evaluate_priorities()

#calculate the Earliness
def EarlinessSum(taskList):
 sumE = 0
 for task in taskList:
     starting_day = date_to_day(task.starting_day)
     due_date = date_to_day(task.due_date)
     E = due_date - starting_day
     if (E < 0): #days after due date
       E = 0
     sumE = sumE + E
 return sumE

