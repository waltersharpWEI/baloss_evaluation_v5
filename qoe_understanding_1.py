#created by Lai Wei at 2020/01/01, 14:47(UTC +8)
#this code is made for drawing the figure.3(b) in BaLoss Sigcomm'21
#a here dentoes Robust-MPC, b denotes Pensieve

#import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum
from qoe_models.qoe_model1 import qoe_formula_normal, qoe_formula_normal_1

class AbrAlgo(Enum):
  """Possible exploration policies."""
  pensieve_algo = 1
  mpc_algo = 2

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

def plot_twin(bitrates_a,bitrates_pensieve,bitrates_mpc,figname):
    # setup the background template of plt
    asize = 28
    #bsize = 35
    lwidth = 5
    xfontsize = 35
    yfontsize = 35
    #afontsize = 24
    #bar_width = 0.02
    #opacity = 0.8
    #bitrate_color = 'navy'
    #bitrate_color_a = 'navy'
    bitrate_color_b = 'red'
    bitrate_color_c = 'navy'
    #bitrate_color_p = 'tab:blue'
    bitrate_color_p = 'black'
    figure = plt.figure(figsize=(16, 9), dpi=80)
    ax = figure.add_axes([0.150, 0.15, 0.7, 0.8])
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(xfontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(yfontsize)
        tick.label1.set_color(bitrate_color_p)
        ax.tick_params(pad=18)
    ax.set_xlabel("Time(s)", fontsize=xfontsize)
    ax.set_ylabel("Bandwidth(Mbps)", fontsize=yfontsize, color=bitrate_color_p)

    #draw the trace1 on fig1's canvas
    lns1a = ax.plot(bitrates_a,color="black",
             linestyle='dashdot',
             label="True Bandwidth",
             linewidth=lwidth)
    lns1b = ax.plot(bitrates_pensieve, color=bitrate_color_b,
                    label="Pensieve",
                    linewidth=lwidth, linestyle='-')
    #lns1c = ax.plot(bitrates_mpc, color=bitrate_color_c,
    #                label="Robust-MPC",
    #                linewidth=lwidth, linestyle='-')



    lns = lns1a + lns1b
    labs = [l.get_label() for l in lns]
    figure.legend(lns, labs, bbox_to_anchor=(0.9, 0.99) , loc='upper right',
                  fontsize=asize, columnspacing=0.5)
    #ax.grid()


    ax.set_yticks([0.4,1.5,4,6,8,10,12,14])
    ax.set_yticklabels([0.4,1.5,4,6,8,10,12,14])

    i = 0
    youtube_level_set = set([0,1,2,4,7])
    youtube_small_level_set = set([0,1])

    for tick in ax.yaxis.get_major_ticks():
        if i in youtube_small_level_set:
            tick.label1.set_fontsize(yfontsize-4)
        else:
            tick.label1.set_fontsize(yfontsize)
        if i in youtube_level_set:
            tick.label1.set_color("red")
        i += 1
        #ax.tick_params(pad=18)
    #ax.set_xlim(0, 100)
    #ax.set_ylim(0, 8)
    #plt.yticks(youtube_bitrate_levels)
    ax.set_xlim(0, 50)
    plt.savefig(figname + ".png")
    plt.show()

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

def plot_twin_loss(loss_a,figname):
    # setup the background template of plt
    asize = 28
    #bsize = 35
    lwidth = 5
    xfontsize = 35
    yfontsize = 35
    #afontsize = 24
    #bar_width = 0.02
    #opacity = 0.8
    #bitrate_color = 'navy'
    #bitrate_color_a = 'navy'
    bitrate_color_b = 'red'
    bitrate_color_c = 'navy'
    #bitrate_color_p = 'tab:blue'
    bitrate_color_p = 'black'
    figure = plt.figure(figsize=(16, 9), dpi=80)
    ax = figure.add_axes([0.150, 0.15, 0.7, 0.8])
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(xfontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(yfontsize)
        tick.label1.set_color(bitrate_color_p)
        ax.tick_params(pad=18)
    ax.set_xlabel("Time(s)", fontsize=xfontsize)
    ax.set_ylabel("Loss(%)", fontsize=yfontsize, color=bitrate_color_p)

    #draw the trace1 on fig1's canvas
    lns1a = ax.plot(loss_a,color="black",
             linestyle='-',
             label="Pensieve Loss",
             linewidth=lwidth)
    #lns1b = ax.plot(bitrates_pensieve, color=bitrate_color_b,
    #                label="Pensieve",
    #                linewidth=lwidth, linestyle='-')
    #lns1c = ax.plot(bitrates_mpc, color=bitrate_color_c,
    #                label="Robust-MPC",
    #                linewidth=lwidth, linestyle='-')



    lns = lns1a
    labs = [l.get_label() for l in lns]
    figure.legend(lns, labs, bbox_to_anchor=(0.9, 0.99) , loc='upper right',
                  fontsize=asize, columnspacing=0.5)
    #ax.grid()


    #ax.set_yticks([0.4,1.5,4,6,8,10,12,14])
    #ax.set_yticklabels([0.4,1.5,4,6,8,10,12,14])

    i = 0


    #ax.set_xlim(0, 100)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 1)
    #plt.yticks(youtube_bitrate_levels)
    plt.savefig(figname + ".png")
    plt.show()


def plot_twin_qoe(qoe_a,figname):
    # setup the background template of plt
    asize = 28
    #bsize = 35
    lwidth = 5
    xfontsize = 35
    yfontsize = 35
    #afontsize = 24
    #bar_width = 0.02
    #opacity = 0.8
    #bitrate_color = 'navy'
    #bitrate_color_a = 'navy'
    bitrate_color_b = 'red'
    bitrate_color_c = 'navy'
    #bitrate_color_p = 'tab:blue'
    bitrate_color_p = 'black'
    figure = plt.figure(figsize=(16, 9), dpi=80)
    ax = figure.add_axes([0.150, 0.15, 0.7, 0.8])
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(xfontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(yfontsize)
        tick.label1.set_color(bitrate_color_p)
        ax.tick_params(pad=18)
    ax.set_xlabel("Time(s)", fontsize=xfontsize)
    ax.set_ylabel("QoE", fontsize=yfontsize, color=bitrate_color_p)

    #draw the trace1 on fig1's canvas
    lns1a = ax.plot(qoe_a,color="black",
             linestyle='-',
             label="Pensieve QoE",
             linewidth=lwidth)
    #lns1b = ax.plot(bitrates_pensieve, color=bitrate_color_b,
    #                label="Pensieve",
    #                linewidth=lwidth, linestyle='-')
    #lns1c = ax.plot(bitrates_mpc, color=bitrate_color_c,
    #                label="Robust-MPC",
    #                linewidth=lwidth, linestyle='-')



    lns = lns1a
    labs = [l.get_label() for l in lns]
    figure.legend(lns, labs, bbox_to_anchor=(0.9, 0.99) , loc='upper right',
                  fontsize=asize, columnspacing=0.5)
    #ax.grid()



    ax.set_xlim(0, 50)
    #ax.set_ylim(0, 8)
    #plt.yticks(youtube_bitrate_levels)
    plt.savefig(figname + ".png")
    plt.show()

def compute_qoe(bitrates, losss):
    i = 0
    delay = 100
    qoes = []
    for bitrate in bitrates:
        loss = losss[i]
        qoes.append(qoe_formula_normal_1(delay,loss,bitrate))
        i+=1
    qoes = np.array(qoes)
    qoes += 2500
    qoes /= qoes.max()
    return qoes

def gen_trace_loss_a_real(length):
    df = pd.read_csv("net_traces/trace1.csv")
    per = df["ber"]
    start = 2000
    trace = per[start:length+start]
    return trace


if __name__=="__main__":
    trace_len = 50
    #bitrates_a = gen_trace_bitrate_a(trace_len)
    #with open("abr_trace.csv",'w') as f1:
    #   bitrates_a.tofile(f1,sep=',')
    #loss_a = np.array(gen_trace_loss_a_real(trace_len))
    #with open("abr_trace_loss.csv",'w') as f1:
    #   loss_a.tofile(f1,sep=',')
    bitrates_a = np.fromfile("abr_trace.csv",sep=',')
    bitrates_pensieve = apply_abr(bitrates_a, algo=AbrAlgo.pensieve_algo)
    bitrates_mpc = apply_abr(bitrates_a, algo=AbrAlgo.mpc_algo)
    loss_a = np.fromfile("abr_trace_loss.csv",sep=',')

    print(loss_a)
    qoe_a = compute_qoe(bitrates_pensieve,loss_a)
    print(qoe_a)
    #bitrates_b = gen_trace_bitrate_b(trace_len)
    plot_twin(bitrates_a,bitrates_pensieve,bitrates_mpc, "qoe_understand1_bandwidth")
    plot_twin_loss(loss_a*500, "qoe_understand1_loss")
    plot_twin_qoe(qoe_a, "qoe_understand1_qoe")