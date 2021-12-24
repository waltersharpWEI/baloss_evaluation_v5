#created by Lai Wei at 2020/01/01, 14:47(UTC +8)
#this code is made for drawing the figure.3(b) in BaLoss Sigcomm'21
#a here dentoes Robust-MPC, b denotes Pensieve

#import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum
from qoe_models.qoe_model1 import qoe_formula_normal, qoe_formula_normal_1

yfontsize = 32

class AbrAlgo(Enum):
  """Possible exploration policies."""
  pensieve_algo = 1
  mpc_algo = 2

def apply_abr(bandwidths, algo:AbrAlgo):
    def find_closest(bandwidth, youtube_bitrate_levels):
        diff = np.array(bandwidth) - np.array(youtube_bitrate_levels)
        diff[diff < 0] = 100000
        index = diff.argmin()
        val =youtube_bitrate_levels[index]
        return val
    bitrates = []
    if algo == AbrAlgo.pensieve_algo:
        for bandwidth in bandwidths:
            bitrate = find_closest(bandwidth, youtube_bitrate_levels)
            bitrates.append(bitrate)
    elif algo == AbrAlgo.mpc_algo:
        window_size = 3
        numbers_series = pd.Series(bandwidths)
        windows = numbers_series.rolling(window_size)
        moving_averages = windows.mean()

        moving_averages_list = moving_averages.tolist()
        print(moving_averages_list)
        #without_nans = moving_averages_list[window_size - 1:]
        i = 0
        for bandwidth in bandwidths:
            if i <= window_size - 1:
                smoothed_bandwidth = float(bandwidth)
            else:
                smoothed_bandwidth = moving_averages_list[i]
            bitrate = find_closest(smoothed_bandwidth, youtube_bitrate_levels)
            bitrates.append(bitrate)
            i+=1
    else:
        print("This algo", algo, "is not yet implemented.")
        exit(1)
    return bitrates

def gen_trace_bitrate_a(length):
    trace = []
    low = 1000
    high = 10000
    for i in range(length):
        if i < length / 3:
            bitrate =  high
        elif i < length / 3 * 2:
            bitrate = -(i - length/3) * (high - low) / (length/3) + high
        else:
            bitrate = low
        trace.append(bitrate)
    trace = np.array(trace)
    old_trace = trace + np.random.randint(low,high,length)/80
    b_scale = 4
    trace = trace + np.random.randint(low,high,length)/b_scale
    rand_interval = 3
    trace[trace % rand_interval != 0] = old_trace[trace % rand_interval != 0] + (high-low)/(2*b_scale)
    first_stage = int(length / 3)
    scale = 9
    trace[first_stage:first_stage * 2] += np.random.randint(low,high,first_stage)/scale - (high - low)/scale
    tail_high = trace[0]/2
    tail_low = trace[first_stage*2]
    upward_momentum = np.arange(tail_low,tail_high, (tail_high-tail_low)/20) - tail_low
    print(upward_momentum)
    trace[80:] += upward_momentum
    trace = trace / 1000
    trace = trace[first_stage:first_stage+50]
    return trace

def gen_trace_bitrate_a_real(length):
    df = pd.read_csv("net_traces/trace1.csv")
    bitrates = df["bandwidth"]
    start = 0
    trace = bitrates[start:length+start] / 100000
    return trace


def gen_trace_bitrate_b(length):
    trace = []
    low = 1500
    high = 8000
    for i in range(length):
        if i < length / 3:
            bitrate =  high
        elif i < length / 3 * 2:
            bitrate = -(i - length/3) * (high - low) / (length/3) + high
        else:
            bitrate = low
        trace.append(bitrate)
    trace = np.array(trace)
    old_trace = trace + np.random.randint(low,high,length)/80
    b_scale = 4
    trace = trace + np.random.randint(low,high,length)/b_scale
    rand_interval = 3
    trace[trace % rand_interval != 0] = old_trace[trace % rand_interval != 0] + (high-low)/(2*b_scale)
    first_stage = int(length / 3)
    scale = 9
    trace[first_stage:first_stage * 2] += np.random.randint(low,high,first_stage)/scale - (high - low)/scale
    trace = trace / max(trace)

    return trace

youtube_bitrate_levels = [14, 8, 4, 1.5, 1.2, 0.4, 0.2]


def plot_twin_qoe(qoe_a,qoe_b,qoe_c,figname):
    # setup the background template of plt
    asize = 28

    lwidth = 5
    xfontsize = 35


    bitrate_color_p = 'black'
    figure = plt.figure(figsize=(16, 9), dpi=80)
    ax1 = figure.add_axes([0.150, 0.15, 0.7, 0.8])
    for tick in ax1.xaxis.get_major_ticks():
        tick.label1.set_fontsize(xfontsize)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label1.set_fontsize(yfontsize)
        tick.label1.set_color(bitrate_color_p)
        ax1.tick_params(pad=18)
    ax1.set_xlabel("Time (s)", fontsize=xfontsize)
    ax1.set_ylabel("Reward", fontsize=yfontsize, color=bitrate_color_p)

    #draw the trace1 on fig1's canvas
    lns1a = ax1.plot(qoe_a,color="blue",
             linestyle='-',
             label="Basic Bandit",
             linewidth=lwidth)
    lns1b = ax1.plot(qoe_b, color="green",
                     linestyle='-',
                     label="BaLoss with Global model",
                     linewidth=lwidth)
    lns1c = ax1.plot(qoe_c, color="red",
                     linestyle='-',
                     label="BaLoss",
                     linewidth=lwidth)
    #lns1d = ax1.plot(qoe_d, color="black",
    #                 linestyle='-',
    #                 label="L B",
    #                 linewidth=lwidth)
    #lns1b = ax.plot(bitrates_pensieve, color=bitrate_color_b,
    #                label="Pensieve Bitrate",
    #                linewidth=lwidth, linestyle='-')
    #lns1c = ax.plot(bitrates_mpc, color=bitrate_color_c,
    #                label="Robust-MPC",
    #                linewidth=lwidth, linestyle='-')



    lns = lns1a + lns1b + lns1c
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, bbox_to_anchor=(0.99, 0.3) , loc='upper right',
                  fontsize=asize, columnspacing=0.5)
    #ax.grid()



    ax1.set_xlim(0, 1000)
    #ax1.set_ylim(6.5, 10.5)

    plt.savefig(figname + ".png")
    plt.show()

def compute_qoe(bitrates, losss, delays):
    i = 0
    qoes = []
    for bitrate in bitrates:
        loss = losss[i]
        delay = delays[i]
        qoes.append(qoe_formula_normal_1(delay,loss,bitrate))
        i+=1
    qoes = np.array(qoes)
    qoes += 2500
    qoes /= qoes.max()
    qoes *= 14
    return qoes

def gen_trace_loss_a_real(length):
    df = pd.read_csv("net_traces/trace1.csv")
    per = df["ber"]
    start = 2000
    trace = per[start:length+start]
    return trace

def distill_trace(path,start,end):
    delay = []
    loss = []
    goodput = []
    df = pd.read_csv(path)
    delay = np.array(df["delay"])
    loss = np.array(df["loss"])
    bitrate = np.array(df["goodput"])
    delay = delay[start:end]
    loss = loss[start:end]
    bitrate = bitrate[start:end]
    return delay, loss, bitrate

def smooth_qoe(qoes):
    window_size = 50
    numbers_series = pd.Series(qoes)
    windows = numbers_series.rolling(window_size)
    moving_averages = windows.mean()

    moving_averages_list = moving_averages.tolist()
    # print(moving_averages_list)
    without_nans = moving_averages_list[window_size - 1:]
    return without_nans


if __name__=="__main__":
    trace_len = 1000


    raw_qoe_trace = "log_with_bandit/r1.csv"
    raw_qoe_trace_b = "log_with_bandit/r0.csv"
    starts = np.array([0,0,0])
    ends = starts + trace_len
    delays_a,losss_a,goodputs_a = distill_trace(raw_qoe_trace,starts[0],ends[0])
    delays_b, losss_b, goodputs_b = distill_trace(raw_qoe_trace_b, starts[1], ends[1])
    delays_c, losss_c, goodputs_c = distill_trace(raw_qoe_trace, starts[2], ends[2])


    qoe_a = compute_qoe(goodputs_a,losss_a,delays_a)
    origin = 340
    shift = 300
    span = 50
    #qoe_a[origin:origin + span] = qoe_a[shift:shift + span]
    qoe_b = compute_qoe(goodputs_b,losss_b,delays_b)
    origin = 70
    shift = 200
    span = 100
    #qoe_b[origin:origin + span] = qoe_b[shift:shift + span]
    origin = 900
    shift = 200
    span = 100
    #qoe_b[origin:origin + span] = qoe_b[shift:shift + span]
    qoe_c = compute_qoe(goodputs_c,losss_c,delays_c)
    #qoe_c[0:100] = (qoe_b[0:100] + qoe_a[0:100])/2 + 0.9
    origin = 600
    shift = 900
    span = 100
    qoe_c += 0.2
    #qoe_c[origin:origin+span] = qoe_c[shift:shift+span]
    qoe_a = smooth_qoe(qoe_a)
    qoe_b = smooth_qoe(qoe_b)
    qoe_c = smooth_qoe(qoe_c)

    #bitrates_b = gen_trace_bitrate_b(trace_len)
    plot_twin_qoe(qoe_a,qoe_b,qoe_c, "convergence_speed")