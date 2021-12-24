#created by Lai Wei at 2020/01/01, 14:47(UTC +8)
#this code is made for drawing the figure.3(b) in BaLoss Sigcomm'21
#a here dentoes Robust-MPC, b denotes Pensieve

#import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum

class AbrAlgo(Enum):
  """Possible exploration policies."""
  pensieve_algo = 1
  mpc_algo = 2


def gen_trace_loss_a(length):
    trace = []
    low = 5000
    high = 10000
    for i in range(length):
        if i < length / 3:
            bitrate =  low
        elif i < length / 3 * 2:
            bitrate = (i - length/3) * (high - low) / (length/3) + low
        else:
            bitrate = high
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
    #print(upward_momentum)
    #trace[80:] += upward_momentum
    trace = np.array(trace) / 100000
    trace[trace < 0] = 0
    return trace

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

def plot_twin(bitrates_a,bitrates_pensieve,bitrates_mpc, loss_pensieve, loss_mpc, figname):
    # setup the background template of plt
    asize = 28
    lwidth = 5
    xfontsize = 35
    yfontsize = 35
    bitrate_color_b = 'navy'
    bitrate_color_c = 'red'
    #bitrate_color_p = 'tab:blue'
    bitrate_color_p = 'black'
    loss_color_p = 'black'
    figure = plt.figure(figsize=(16, 9), dpi=80)
    ax = figure.add_axes([0.150, 0.15, 0.7, 0.8])
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(xfontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(yfontsize)
        tick.label1.set_color(bitrate_color_p)
        ax.tick_params(pad=18)
    ax.set_xlabel("Time(s)", fontsize=xfontsize)
    ax.set_ylabel("Bitrate(Mbps)", fontsize=yfontsize, color=bitrate_color_p)

    #draw the trace1 on fig1's canvas
    lns1a = ax.plot(bitrates_a,color="black",
             #linestyle='dashdot',
             label="True Bandwidth",
             linewidth=lwidth)
    lns1b = ax.plot(bitrates_pensieve, color=bitrate_color_b,
                    label="Pensieve",
                    linewidth=lwidth, linestyle='-')
    lns1c = ax.plot(bitrates_mpc, color=bitrate_color_c,
                    label="Robust-MPC",
                    linewidth=lwidth, linestyle='-')




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

    # create a twinx on ax
    ax2 = ax.twinx()
    ax2.set_ylabel("UDP Loss(%)", fontsize=yfontsize, color=loss_color_p)
    for tick in ax2.xaxis.get_major_ticks():
        tick.label2.set_fontsize(xfontsize)
    for tick in ax2.yaxis.get_major_ticks():
        tick.label2.set_fontsize(yfontsize)
        tick.label2.set_color(loss_color_p)
        ax2.tick_params(pad=18)



    print(loss_pensieve)
    lns2a = ax2.plot(loss_pensieve, color='blue',
                    label="Pensieve",
                    linewidth=lwidth, linestyle='dotted')
    lns2b = ax2.plot(loss_mpc, color='red',
                    label="Robust-MPC",
                    linewidth=lwidth, linestyle='dotted')

    lns = lns1a + lns1b + lns1c
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='upper left',
                  fontsize=asize, columnspacing=0.5)
    lns2 = lns2a + lns2b
    labs2 = [l.get_label() for l in lns2]
    ax2.legend(lns2, labs2,  loc='center right',
                  fontsize=asize, columnspacing=0.5)
    #ax2.legend(loc=0)

    #ax.set_ylim(0, 8)
    #plt.yticks(youtube_bitrate_levels)
    ax.set_xlim(0, 100)
    ax.set_ylim(0,20)

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
            bitrates.append(bitrate-1)
            i+=1
    else:
        print("This algo", algo, "is not yet implemented.")
        exit(1)
    return bitrates

if __name__=="__main__":
    trace_len = 100
    #bitrates_a = gen_trace_bitrate_a(trace_len)
    #with open("abr_trace.csv",'w') as f1:
    #    bitrates_a.tofile(f1,sep=',')
    bitrates_a = np.fromfile("abr_trace.csv",sep=',')
    bitrates_pensieve = apply_abr(bitrates_a, algo=AbrAlgo.pensieve_algo)
    bitrates_mpc = apply_abr(bitrates_a, algo=AbrAlgo.mpc_algo)
    loss_pensieve = gen_trace_loss_a(100)
    loss_mpc = gen_trace_loss_a(100)
    bitrate_util_pensieve = (np.array(bitrates_pensieve) / np.array(bitrates_a)).mean()
    bitrate_util_mpc = (np.array(bitrates_mpc) / np.array(bitrates_a)).mean()
    mean_loss_pensieve = loss_pensieve.mean()
    mean_loss_mpc = loss_mpc.mean()
    print("bitrate utilization(pensieve):",bitrate_util_pensieve)
    print("bitrate utilization(mpc):", bitrate_util_mpc)
    print("avrage loss(pensieve):",mean_loss_pensieve)
    print("avrage loss(mpc):", mean_loss_mpc)
    #print(loss_pensieve)
    #bitrates_b = gen_trace_bitrate_b(trace_len)
    plot_twin(bitrates_a,bitrates_pensieve,bitrates_mpc,
              loss_pensieve, loss_mpc, "abr_time_step_gain")