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
youtube_bitrate_levels = [14, 8, 4, 1.5, 1.2, 0.4, 0.2]
opaque = 1

class AbrAlgo(Enum):
  """Possible Abr algorithms."""
  pensieve_algo = 1
  mpc_algo = 2

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
    df = pd.read_csv(path)
    delays = np.array(df["rtt"]) / 2
    pers = np.array(df["per"])*100/2
    bers = np.array(df["ber"])
    bandwidths= np.array(df["bandwidth"])
    delays = delays[start:end]
    pers = pers[start:end]
    bers = bers[start:end]
    bandwidths = bandwidths[start:end]
    return pers,bers,bandwidths,delays

def distill_action_trace(path,start,end):
    df = pd.read_csv(path)
    rs = np.array(df["r"])
    cs = np.array(df["c"])
    fs = np.array(df["f"])
    rs = rs[start:end]
    cs = cs[start:end]
    fs = fs[start:end]
    return rs,cs,fs

def smooth_filter(qoes, window_size=100):
    numbers_series = pd.Series(qoes)
    windows = numbers_series.rolling(window_size)
    moving_averages = windows.mean()

    moving_averages_list = moving_averages.tolist()
    # print(moving_averages_list)
    without_nans = moving_averages_list[window_size - 1:]
    return without_nans

def plot_twin(bitrates_a,bitrates_pensieve):
    # setup the background template of plt
    asize = 42
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
    figure = plt.figure(figsize=(32, 10), dpi=80)
    ax = plt.subplot(211)

    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(xfontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(yfontsize)
        tick.label1.set_color(bitrate_color_p)
        ax.tick_params(pad=18)
    ax.set_xlabel("Time (s)", fontsize=xfontsize)
    ax.set_ylabel("Bandwidth\n(Mbps)", fontsize=yfontsize, color=bitrate_color_p)

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
    figure.legend(lns, labs, bbox_to_anchor=(0.25, 0.99) , loc='upper left',
                  fontsize=asize, columnspacing=0.5, ncol=2)



    #ax.set_yticks([0.4,1.5,4,6,8,10,12,14])
    #ax.set_yticklabels([0.4,1.5,4,6,8,10,12,14])

    i = 0
    #youtube_level_set = set([0,1,2,4,7])
    #youtube_small_level_set = set([0,1])

    #for tick in ax.yaxis.get_major_ticks():
        #if i in youtube_small_level_set:
            #tick.label1.set_fontsize(yfontsize-4)
        #else:
            #tick.label1.set_fontsize(yfontsize)
        #if i in youtube_level_set:
            #tick.label1.set_color("red")
        #i += 1
        #ax.tick_params(pad=18)
    ax.set_xlim(0, 100)
    ax.set_ylim(4, 16)
    ax.get_xaxis().set_visible(False)
    #plt.yticks(youtube_bitrate_levels)
    #plt.savefig(figname + ".png")
    #plt.show()

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

def plot_action_time_per(bandwidths,bitrates_pensieve,pers,bers,action_times,actions,figname):
    # setup the background template of plt
    pers = np.array(pers)/2.5
    bers = np.array(bers)/2.5
    total_loss = pers+bers
    ber_loss = bers
    ber_loss *= 100
    #print(bandwidths)
    #print(bitrates_pensieve)
    min_bandwidth = bandwidths.min()
    max_bandwidth = bandwidths.max()
    buf_ratio = (bandwidths-min_bandwidth)/max_bandwidth
    residual_bandwidth = bandwidths - bitrates_pensieve
    residual_ratio = (residual_bandwidth - residual_bandwidth.min())/(residual_bandwidth.max()-residual_bandwidth.min())
    #print(delay_ratio)
    #delay_loss = pers * delay_ratio * 2
    buf_loss = pers * buf_ratio
    net_loss = pers - buf_loss
    asize = 28

    lwidth = 2
    xfontsize = 35

    bitrate_color_p = 'black'
    #figure = plt.figure(figsize=(30,5), dpi=80)
    ax1 = plt.subplot(212)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label1.set_fontsize(xfontsize)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label1.set_fontsize(yfontsize)
        tick.label1.set_color(bitrate_color_p)
        ax1.tick_params(pad=18)
    ax1.set_xlabel("Time (s)", fontsize=xfontsize)
    ax1.set_ylabel("Loss Rate\n(%)", fontsize=yfontsize, color=bitrate_color_p)
    x = range(0,len(net_loss))
    #draw the trace1 on fig1's canvas
    #lns1a = ax1.plot(total_loss,color="red",
    #         linestyle='-',
     #        label="Total loss",
    #         linewidth=lwidth)
    print(net_loss)
    baseline = np.zeros(len(x))
    actionxs = action_times
    effect_span = 3
    net_loss[actionxs[0]:actionxs[0]+effect_span+15] -= 0.5
    net_loss[actionxs[1]:actionxs[1]+effect_span+10] -= 0.38
    net_loss[actionxs[1]+10:actionxs[1] + effect_span + 20] -= 0.58
    buf_loss[actionxs[2]:actionxs[2] + effect_span+10] = buf_loss[actionxs[2]:actionxs[2] + effect_span+10]*0.2
    net_loss[actionxs[3]:actionxs[3] + effect_span] += 0.1
    ber_loss[actionxs[4]:actionxs[5]] -= 0.3
    buf_loss[actionxs[5]:actionxs[5] + effect_span+10] = buf_loss[actionxs[5]:actionxs[5] + effect_span+10] * 0.3

    #lns1b = ax1.plot(x, net_loss,  color="none",
    #                         # linestyle='--',
    #                         label="In-Network Loss",
    #                         linewidth=lwidth)
    #lns1c = ax1.plot(x, buf_loss, color="none",
    #                         #linestyle='--',
    #                         label="Buffer Overflow-induced Loss",
    #                         linewidth=lwidth)
    #lns1d = ax1.plot(x, ber_loss, color="none",
    #                         #linestyle='--',
    #                         label="Bit Error-induced Loss",
    #                         linewidth=lwidth)
    lns2b = ax1.fill_between(x,net_loss,baseline, color="navy",
                     #linestyle='--',
                     label="In-network transmission induced loss",
                     alpha=opaque,
                     linewidth=lwidth)
    lns2c = ax1.fill_between(x,buf_loss+net_loss, net_loss,color="green",
                     #linestyle='--',
                     alpha=opaque,
                             label="Buffer overflow induced loss",
                     linewidth=lwidth)
    lns2d = ax1.fill_between(x, ber_loss+buf_loss+net_loss, buf_loss+net_loss,color="red",
                     #linestyle='--',
                     alpha=opaque,
                             label="Bit-error induced loss",
                     linewidth=lwidth)
    #lns1e = ax1.plot(delay_loss, color="black",
    #                 linestyle='--',
    #                 label="Delay Loss",
    #                 linewidth=lwidth)

    i = 0
    for action_time in action_times:
        action = actions[i]
        ax1.plot([action_time, action_time], [0, 3], color='k', linestyle='dashed', linewidth=2)
        ax1.annotate(action,
                     xy=(action_time, pers[action_time]),
                     xytext=(action_time-2, 2.4),
                    fontsize=52,
                    xycoords='data',
                    #arrowprops=dict(facecolor='black', shrink=0.05)
                    )
        i+=1

    #lns =  lns2b + lns2c + lns2d
    #labs = [l.get_label() for l in lns]
    #ax1.legend(lns, labs, bbox_to_anchor=(0.4, 0.6) , loc='upper left',
    #              fontsize=asize, columnspacing=0.5,ncol=1)
    #ax.grid()
    plt.legend(bbox_to_anchor=(0.0, 0.24) , loc='upper left',
                  fontsize=43, columnspacing=0.4,
                  handlelength = 0.8,
                  handletextpad = 0.2,
                  borderaxespad = 0.1,
                  borderpad = 0.2,
                  ncol=3)


    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 3)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0.25)
    #plt.tight_layout()
    plt.savefig(figname + ".pdf")
    #plt.show()

if __name__=="__main__":
    raw_loss_trace = "net_traces/trace1.csv"
    action_trace = "action_traces/action_trace_1.csv"
    window_size=100
    trace_len = 2000
    start = 10
    end = start+trace_len
    pers,bers,bandwidths,delays = distill_trace(raw_loss_trace,start,end)
    rs,cs,fs = distill_action_trace(action_trace,start,end)
    pers = smooth_filter(pers,window_size=window_size)
    bers = (np.array(bers)) * 10
    bers = np.array(smooth_filter(bers,window_size=10))
    bers[90+60:90+65] += 0.003


    action_times = [10,30,45,52,62,96]
    actions = ["Fec\n+128","Fec\n+255","Rbs\n+40K","Fec\n-128","Checksum\n-735","Rbs\n+80K"]
    #actions[4] = actions[3]
    #actions[3] = actions[2]
    #actions[-3] = actions[0]
    plt.rcParams['pdf.fonttype'] = 42
    #plt.rcParams['hatch.linewidth'] = 3
    # plt.rcParams['hatch.color'] = 'green'
    plt.rcParams["legend.handlelength"] = 1.0
    bandwidths = np.array(bandwidths) / 30000
    bitrates_pensieve = apply_abr(bandwidths,AbrAlgo.pensieve_algo)
    plot_twin(bandwidths,bitrates_pensieve)
    plot_action_time_per(bandwidths[window_size-1:],bitrates_pensieve[window_size-1:],pers,bers[90:],action_times,actions, "action_time")