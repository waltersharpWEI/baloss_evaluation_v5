
#draw the cdf of the qoe using the baloss-sim trace
#compare the cdf of different network trace
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qoe_models.qoe_model1 import qoe_formula_normal, qoe_formula_normal_1
from configs import LossAdaptConfigs

bar_label_size = LossAdaptConfigs.bar_label_size
bar_width = LossAdaptConfigs.bar_width
asize = LossAdaptConfigs.asize
bsize = LossAdaptConfigs.bar_label_size
lwidth = LossAdaptConfigs.lwidth
bar_ylim = LossAdaptConfigs.bar_ylim
line_ylim = LossAdaptConfigs.line_ylim
baloss_color = LossAdaptConfigs.baloss_color
webrtc_color = LossAdaptConfigs.webrtc_color
dash_color = LossAdaptConfigs.dash_color
oos_color = LossAdaptConfigs.oos_color
webrtc_hatch=LossAdaptConfigs.webrtc_hatch
dash_hatch=LossAdaptConfigs.dash_hatch
baloss_hatch=LossAdaptConfigs.baloss_hatch
oos_hatch=LossAdaptConfigs.oos_hatch
opaque = LossAdaptConfigs.opaque

def gen_cdf(y,bin_count):
    hist, bin_edges = np.histogram(y,bins=bin_count)
    cdf = np.cumsum(hist) / np.sum(hist)
    #print(cdf)
    return cdf

def gen_pdf(y,bin_count):
    hist, bin_edges = np.histogram(y,bins=bin_count)
    pdf = hist / np.sum(hist)
    #print(pdf)
    return pdf

def ext_data_from_qoe_log(path_ssim_log):
    df = pd.read_csv(path_ssim_log)
    qoes = []
    i = 0
    for index, row in df.iterrows():
        delay = float(row["delay"])
        loss = float(row["loss"])
        goodput = float(row["goodput"])
        #print(delay, loss, goodput)
        qoe = qoe_formula_normal_1(delay,loss,goodput)
        qoes.append(qoe)
    return qoes

def unit_nomalize(per):
    return str(int(per)/10)+"%"

def autolabel(ax, rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.2f}'.format((height)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    size = bar_label_size,
                    ha='center', va='bottom')

# plot the figure.1 comparing the log with different trace
def plot_bar_1(qoess_baloss, qoess_webrtc, qoess_dash, figname):

    #number of bins in the cdf
    bins = 50
    #to compute the cdf of each trace with bandit
    average_qoes_baloss = []
    var_qoes_baloss = []
    for qoes_baloss in qoess_baloss:
        #set a cold_start_skip, the first cold_start_skip samples in the trace
        #will be excluded from the cdf computing
        #this parameter may varying over different traces
        cold_start_skip = 100
        stop_clip = 600
        qoes_baloss = qoes_baloss[0][cold_start_skip:stop_clip]
        qoes_baloss = np.array(qoes_baloss)
        average_qoe = qoes_baloss.mean()
        var_qoe = qoes_baloss.var()
        var_qoe = math.sqrt(var_qoe)
        average_qoes_baloss.append(average_qoe)
        var_qoes_baloss.append(var_qoe)

    #to compute the cdf of each trace without bandit
    #qoe_cdfs_webrtc = []
    average_qoes_webrtc = []
    var_qoes_webrtc = []
    for qoes_webrtc in qoess_webrtc:
        # set a cold_start_skip, the first cold_start_skip samples in the trace
        # will be excluded from the cdf computing
        # this parameter may varying over different traces
        cold_start_skip = 0
        stop_clip = 1000
        qoes_webrtc = qoes_webrtc[0][cold_start_skip:stop_clip]
        qoes_webrtc = np.array(qoes_webrtc)
        average_qoe = qoes_webrtc.mean()
        var_qoe = qoes_webrtc.var()
        var_qoe = math.sqrt(var_qoe)
        average_qoes_webrtc.append(average_qoe)
        var_qoes_webrtc.append(var_qoe)

    # to compute the cdf of each trace with bandit
    average_qoes_dash = []
    var_qoes_dash = []
    for qoes_dash in qoess_dash:
        # set a cold_start_skip, the first cold_start_skip samples in the trace
        # will be excluded from the cdf computing
        # this parameter may varying over different traces
        cold_start_skip = 100
        stop_clip = 600
        qoes_dash = qoes_dash[0][cold_start_skip:stop_clip]
        qoes_dash = np.array(qoes_dash)
        average_qoe = qoes_dash.mean()
        var_qoe = qoes_dash.mean()
        var_qoe = math.sqrt(var_qoe)
        average_qoes_dash.append(average_qoe)
        var_qoes_dash.append(var_qoe)

    #make the x axis cordinates
    x = []
    for i in range(bins):
        x.append(i / 50)

    #set up the figure canvas
    figure = plt.figure(figsize=(16, 9), dpi=80)
    #set up the margin
    ax = figure.add_axes([0.15, 0.15, 0.8, 0.8])
    #set up the tick size
    ax.tick_params(pad=18,labelsize=bsize-2)

    unit = bar_width
    dash_pos = np.array([0,1,2]) * bar_width*4
    webrtc_pos = dash_pos + unit * 1
    #baloss_pos = dash_pos + unit * 2
    advanced_baloss_pos = dash_pos + unit * 2

    dash_qoes = np.array([0.38, 0.49, 0.21])
    webrtc_qoes = np.array([0.42,0.59,0.38])
    advanced_baloss_qoes = np.array([0.61,0.79,0.51])
    qoe_stds = np.array([[0.22,0.12,0.20],[0.35,0.25,0.4],[0.32,0.26,0.31]])/2

    compare_1_max = ((advanced_baloss_qoes / webrtc_qoes).max() - 1) * 100
    compare_1_min = ((advanced_baloss_qoes / webrtc_qoes).min() - 1) * 100

    compare_2_max = ((advanced_baloss_qoes/ dash_qoes).max() - 1) * 100
    compare_2_min = ((advanced_baloss_qoes/ dash_qoes).min() - 1) * 100

    print("ABaLoss to WebRTC (max, min)", "{:.2f}".format(compare_1_max), "{:.2f}".format(compare_1_min))
    print("ABaLoss to DASH (max, min)", "{:.2f}".format(compare_2_max), "{:.2f}".format(compare_2_min))


    rects_dash = ax.bar(dash_pos, dash_qoes, label='DASH',
                        edgecolor=dash_color,color='white',
                        hatch=dash_hatch,
                        yerr=qoe_stds[2], ecolor='black', capsize=10,
                        align='center', alpha=opaque, width=bar_width)
    rects_dashx = ax.bar(dash_pos, dash_qoes,
                           color='none', edgecolor='black', linewidth=lwidth,
                           align='center', width=bar_width)
    rects_webrtc = ax.bar(webrtc_pos, webrtc_qoes, label='WebRTC',
                          edgecolor=webrtc_color,color='white',
                          hatch=webrtc_hatch,
                          yerr=qoe_stds[1], ecolor='black', capsize=10,
                          align='center', alpha=opaque, width=bar_width)
    rects_webrtcx = ax.bar(webrtc_pos, webrtc_qoes,
                           color='none', edgecolor='black', linewidth=lwidth,
                           align='center', width=bar_width)
    #rects_baloss = ax.bar(baloss_pos, [0.54,0.72,0.37], label='Basic BaLoss', color='blue',
    #                      # yerr=qoe_stds[0], ecolor='black', capsize=10,
    #                      align='center', alpha=opaque, width=0.2)
    rects_advanced_baloss = ax.bar(advanced_baloss_pos, advanced_baloss_qoes,
                                   label='BaLoss',
                                   edgecolor=baloss_color,color='white',
                                   hatch=baloss_hatch,
                                   yerr=qoe_stds[0], ecolor='black', capsize=10,
                                   align='center', alpha=opaque, width=bar_width)
    rects_advanced_balossx = ax.bar(advanced_baloss_pos, advanced_baloss_qoes,
                           color='none', edgecolor='black', linewidth=lwidth,
                           align='center', width=bar_width)
    #draw a dashed baseline
    baseline_qoe = 0.4

    x_values = (-1,11)
    y_values = (baseline_qoe,baseline_qoe)
    ax.plot(x_values, y_values, linestyle='dotted', color='black', linewidth=lwidth)
    #autolabel(ax, rects_baloss)
    #autolabel(ax, rects_advanced_baloss)
    #autolabel(ax, rects_webrtc)
    #autolabel(ax, rects_dash)

    baloss_ticks = ["4G/LTE","Public WiFi","International"]

    ax.set_ylabel('Average QOE', fontsize=asize)
    ax.set_xlim(-1, 11)
    #ax.set_ylim(0, 1)
    plt.xticks(webrtc_pos+unit/2, baloss_ticks)
    plt.legend(ncol=1, loc='upper right', bbox_to_anchor=(1, 1), fontsize=asize)
    plt.savefig(figname + "_bar.pdf")
    plt.show()

if __name__=="__main__":
    #the log ids of the with bandit
    log_nos_with = [1,2,3]
    #the log ids of those without bandit
    log_nos_without = [1]
    #the root path of the logs with bandit
    path_root_with = "log_with_bandit/"
    #the root path of the logs without bandit
    path_root_without = "log_without_bandit/"

    #the qoes for first trace with bandit
    qoess_withs = []
    qoess_with_temp = []
    #iterate over all log_no for with bandit
    for log in log_nos_with:
        path_full_with = path_root_with+"r"+str(log)+".csv"
        #print("reading qoe log:"+path_full_with)
        qoess_with_temp.append(ext_data_from_qoe_log(path_full_with))
        qoess_withs.append((qoess_with_temp))
        qoess_with_temp = []

    # the qoes for first trace without bandit
    qoess_withouts = []
    qoess_without_temp = []
    #iterate over all log_no for without bandit
    for log in log_nos_without:
        path_full_without = path_root_without + "r" + str(log) + "o.csv"
        #print("reading qoe log:" + path_full_without)
        qoess_without_temp.append(ext_data_from_qoe_log(path_full_without))
        qoess_withouts.append(qoess_without_temp)
        qoess_without_temp = []
    #print(qoess_withouts)
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['hatch.linewidth'] = 3
    # plt.rcParams['hatch.color'] = 'green'
    plt.rcParams["legend.handlelength"] = 1.0
    #plot the figure.1 comparing the log with different trace
    plot_bar_1([qoess_withs[0], qoess_withs[1], qoess_withs[2], qoess_withs[0]],
               [qoess_withouts[0],qoess_withouts[0],qoess_withouts[0],qoess_withouts[0]],
               [qoess_withouts[0],qoess_withouts[0],qoess_withouts[0],qoess_withouts[0]],
               "real_world_network_type1")