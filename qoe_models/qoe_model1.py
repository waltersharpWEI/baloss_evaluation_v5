from qoe_models.qoex1 import qoe_1

def qoe_formula_normal(delay, loss, goodput):
    val = qoe_1(delay, loss, goodput)
    if val < 0:
        val = 0
    return val

def qoe_formula_normal_1(delay, loss, goodput):
    a = 100
    b = -10000
    c = 2
    if delay == 0:
        delay = 10
    return a * 1/delay + b * loss + c * goodput