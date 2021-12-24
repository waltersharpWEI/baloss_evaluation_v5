#Created by , , on 2020/12/15 11:15 (UTC+8)
#Provides the QoE formulas
import math


def g(l):
    #400 500 optimal
    #250 200 optimal
    th = 10
    l = abs(l)
    y=1/(1+math.exp(th-l))-1/(1+math.exp(th))
    return y

def qoe1(r,l,R):
    #buffer tuning 1,30
    #udplite 5,1
    a1 = 1
    a2 = 30
    R0 = 100
    if r > R:
        r = R
    q = a1 * math.log(r/R0) - a2 * g(l)
    return q

def qoex1(delay, SBR, PER):
    r = SBR * (1 - PER)
    q = qoe1(r, delay, r/10) + 5
    return q

def qoe(FR, SBR, PER):
    a = [1,0.1,1,0,0]
    q = (a[0] + a[1] * FR + a[2]*math.log(SBR)/(1+a[3]*PER+a[4]*PER*PER))
    return q


def qoex(delay, SBR, PER):
    c = 1
    FR = c / (delay/1000)
    #print(FR)
    q = qoe(FR, SBR, PER)
    return q


def qoe_1(delay,loss,goodput):
    val = qoex1(delay,goodput,loss)
    if val < 0:
        val = 0
    return val