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

def plot_twin(bitrates_a,bitrates_pensieve,bitrates_mpc,figname):
    # setup the background template of plt
    asize = 28
    lwidth = 5
    xfontsize = 35

    bitrate_color_b = 'red'
    bitrate_color_p = 'black'
    #ax = figure.add_axes([0.150, 0.15, 0.7, 0.8])
    ax1 = plt.subplot(312)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label1.set_fontsize(xfontsize)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label1.set_fontsize(yfontsize)
        tick.label1.set_color(bitrate_color_p)
        ax1.tick_params(pad=18)
    ax1.set_xlabel("Time(s)", fontsize=xfontsize)
    ax1.set_ylabel("Bandwidth\n(Mbps)", fontsize=yfontsize, color=bitrate_color_p)

    #draw the trace1 on fig1's canvas
    lns1a = ax1.plot(bitrates_a,color="black",
             linestyle='dashdot',
             label="True Bandwidth",
             linewidth=lwidth)
    lns1b = ax1.plot(bitrates_pensieve, color=bitrate_color_b,
                    label="Pensieve Bitrate",
                    linewidth=lwidth, linestyle='-')
    #lns1c = ax.plot(bitrates_mpc, color=bitrate_color_c,
    #                label="Robust-MPC",
    #                linewidth=lwidth, linestyle='-')



    lns = lns1a + lns1b
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, bbox_to_anchor=(0.99, 0.25) , loc='upper right',
                  fontsize=asize, columnspacing=0.5, ncol=2)
    #ax.grid()


    ax1.set_yticks([1.5,4,8,14])
    ax1.set_yticklabels([1.5,4,8,14])

    i = 0
    youtube_level_set = set([0,1,2,4,7])
    youtube_small_level_set = set([0,1])

    for tick in ax1.yaxis.get_major_ticks():
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
    ax1.set_xlim(0, 50)
    ax1.get_xaxis().set_visible(False)




def plot_twin_loss(loss_a,figname):
    # setup the background template of plt
    asize = 28
    lwidth = 5
    xfontsize = 35

    bitrate_color_b = 'red'
    bitrate_color_c = 'navy'
    #bitrate_color_p = 'tab:blue'
    bitrate_color_p = 'black'
    figure = plt.figure(figsize=(12, 16), dpi=80)
    ax2 = plt.subplot(311)
    for tick in ax2.xaxis.get_major_ticks():
        tick.label1.set_fontsize(xfontsize)
    for tick in ax2.yaxis.get_major_ticks():
        tick.label1.set_fontsize(yfontsize)
        tick.label1.set_color(bitrate_color_p)
        ax2.tick_params(pad=18)
    ax2.set_xlabel("Time(s)", fontsize=xfontsize)
    ax2.set_ylabel("Loss(%)", fontsize=yfontsize, color=bitrate_color_p)

    #draw the trace1 on fig1's canvas
    lns1a = ax2.plot(loss_a,color="red",
             linestyle='dotted',
             label="Pensieve",
             linewidth=lwidth)
    #lns1b = ax.plot(bitrates_pensieve, color=bitrate_color_b,
    #               label="Pensieve",
    #                linewidth=lwidth, linestyle='-')
    #lns1c = ax.plot(bitrates_mpc, color=bitrate_color_c,
    #                label="Robust-MPC",
    #                linewidth=lwidth, linestyle='-')



    lns = lns1a
    labs = [l.get_label() for l in lns]
    #ax2.legend(lns, labs, bbox_to_anchor=(0.9, 0.99) , loc='upper right',
    #              fontsize=asize, columnspacing=0.5)
    #ax.grid()


    #ax.set_yticks([0.4,1.5,4,6,8,10,12,14])
    #ax.set_yticklabels([0.4,1.5,4,6,8,10,12,14])

    i = 0


    #ax.set_xlim(0, 100)
    ax2.set_xlim(0, 50)
    ax2.set_ylim(0, 80)
    #plt.yticks(youtube_bitrate_levels)
    ax2.get_xaxis().set_visible(False)



def plot_twin_qoe(qoe_a,qoe_b,qoe_c,qoe_d,figname):
    # setup the background template of plt
    asize = 28

    lwidth = 5
    xfontsize = 35

    #afontsize = 24
    #bar_width = 0.02
    #opacity = 0.8
    #bitrate_color = 'navy'
    #bitrate_color_a = 'navy'
    bitrate_color_b = 'red'
    bitrate_color_c = 'navy'
    #bitrate_color_p = 'tab:blue'
    bitrate_color_p = 'black'
    #figure = plt.figure(figsize=(16, 4), dpi=80)
    ax3 = plt.subplot(313)
    for tick in ax3.xaxis.get_major_ticks():
        tick.label1.set_fontsize(xfontsize)
    for tick in ax3.yaxis.get_major_ticks():
        tick.label1.set_fontsize(yfontsize)
        tick.label1.set_color(bitrate_color_p)
        ax3.tick_params(pad=18)
    ax3.set_xlabel("Time(s)", fontsize=xfontsize)
    ax3.set_ylabel("Realtime-QoE", fontsize=yfontsize, color=bitrate_color_p)

    #draw the trace1 on fig1's canvas
    lns1a = ax3.plot(qoe_a,color="blue",
             linestyle='-',
             label="NOL NOB",
             linewidth=lwidth)
    lns1b = ax3.plot(qoe_b, color="green",
                     linestyle='-',
                     label="NOL B",
                     linewidth=lwidth)
    lns1c = ax3.plot(qoe_c, color="red",
                     linestyle='-',
                     label="L NOB",
                     linewidth=lwidth)
    lns1d = ax3.plot(qoe_d, color="black",
                     linestyle='-',
                     label="L B",
                     linewidth=lwidth)
    #lns1b = ax.plot(bitrates_pensieve, color=bitrate_color_b,
    #                label="Pensieve Bitrate",
    #                linewidth=lwidth, linestyle='-')
    #lns1c = ax.plot(bitrates_mpc, color=bitrate_color_c,
    #                label="Robust-MPC",
    #                linewidth=lwidth, linestyle='-')



    lns = lns1a + lns1b + lns1c + lns1d
    labs = [l.get_label() for l in lns]
    ax3.legend(lns, labs, bbox_to_anchor=(0.4, 0.8) , loc='upper right',
                  fontsize=asize, columnspacing=0.5)
    #ax.grid()



    ax3.set_xlim(0, 50)
    #ax.set_ylim(0, 8)
    #plt.yticks(youtube_bitrate_levels)
    plt.subplots_adjust(wspace=0, hspace=0.2)  # 调整子图间距
    plt.tight_layout()
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
    #with open("abr_trace2.csv",'w') as f1:
    #   bitrates_a.tofile(f1,sep=',')
    #loss_a = np.array(gen_trace_loss_a_real(trace_len))
    #with open("abr_trace_loss.csv",'w') as f1:
    #   loss_a.tofile(f1,sep=',')
    bitrates_a = np.fromfile("abr_trace2.csv",sep=',')
    bitrates_a = bitrates_a[:trace_len]
    bitrates_pensieve = apply_abr(bitrates_a, algo=AbrAlgo.pensieve_algo)
    bitrates_mpc = apply_abr(bitrates_a, algo=AbrAlgo.mpc_algo)
    loss_a = np.fromfile("abr_trace_loss.csv",sep=',')
    loss_a = loss_a[:trace_len]
    loss_trend = np.arange(0, 0.0050, 0.0001)
    loss_a[:50] += loss_trend


    fec_recover_limits = (bitrates_a - bitrates_pensieve) / bitrates_pensieve
    loss_b = loss_a - fec_recover_limits * 0.01
    loss_b[loss_b < 0] = 0
    bitrates_a_p = bitrates_a
    qoe_a = compute_qoe(bitrates_a_p,loss_a)
    qoe_b = compute_qoe(bitrates_pensieve,loss_a)
    qoe_c = compute_qoe(bitrates_a_p,loss_b)
    qoe_d = compute_qoe(bitrates_pensieve,loss_b)

    #bitrates_b = gen_trace_bitrate_b(trace_len)
    plot_twin_loss(loss_a * 5000, "qoe_understand1_loss")
    plot_twin(bitrates_a,bitrates_pensieve,bitrates_mpc, "qoe_understand1_bandwidth")
    plot_twin_qoe(qoe_a,qoe_b,qoe_c,qoe_d, "qoe_understand_loss1c")