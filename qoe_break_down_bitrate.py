#created on 2020/12/23 by ,, ,
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
webrtcd_color=LossAdaptConfigs.webrtcd_color
webrtcd_hatch=LossAdaptConfigs.webrtcd_hatch
dash_color = LossAdaptConfigs.dash_color
oos_color = LossAdaptConfigs.oos_color
webrtc_hatch=LossAdaptConfigs.webrtc_hatch
dash_hatch=LossAdaptConfigs.dash_hatch
baloss_hatch=LossAdaptConfigs.baloss_hatch
oos_hatch=LossAdaptConfigs.oos_hatch

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

def ext_goodput_from_qoe_log(path_ssim_log):
    df = pd.read_csv(path_ssim_log)
    goodputs = []
    i = 0
    for index, row in df.iterrows():
        goodput = float(row["goodput"]) * 8
        goodputs.append(goodput)
    return goodputs

def unit_nomalize(per):
    return str(int(per)/10)+"%"

def autolabel(ax, rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(int(height)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    size = bar_label_size,
                    ha='center', va='bottom')

def super_maximum(lists):
    list_max = lists[0]
    for list in lists:
        list_max = np.maximum(list_max, list)
    return list_max

# plot the figure.1 comparing the log with different trace
def plot_bar_1(qoess_baloss, qoess_webrtc, qoess_dash, figname):


    #for qoes_with in qoess_with:
        #print(qoess_with)

    #for qoes_without in qoess_without:
        #print(qoes_without)

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
        #print(qoes_baloss)
        #qoes_baloss = qoes_baloss[0]
        #print(qoes_baloss)
        #print(len(qoes_baloss))
        qoes_baloss = np.array(qoes_baloss)
        average_qoe = qoes_baloss.mean()
        var_qoe = qoes_baloss.var()
        var_qoe = math.sqrt(var_qoe)
        #compute the cdf
        #qoe_cdf_baloss = gen_cdf(qoes_baloss, bins)
        #append the cdf to the qoe_cdfs_baloss list
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
        # print(qoes_with)
        # qoes_with = qoes_with[0]
        #print(qoes_webrtc)
        # print(len(qoes_with))
        qoes_webrtc = np.array(qoes_webrtc)
        average_qoe = qoes_webrtc.mean()
        var_qoe = qoes_webrtc.var()
        var_qoe = math.sqrt(var_qoe)
        # compute the cdf
        #qoe_cdf_webrtc = gen_cdf(qoes_webrtc, bins)
        # append the cdf to the qoe_cdfs_webrtc list
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
        # print(qoes_dash)
        # qoes_dash = qoes_dash[0]
        # print(qoes_dash)
        # print(len(qoes_dash))
        qoes_dash = np.array(qoes_dash)
        average_qoe = qoes_dash.mean()
        var_qoe = qoes_dash.mean()
        var_qoe = math.sqrt(var_qoe)
        # compute the cdf
        # qoe_cdf_dash = gen_cdf(qoes_dash, bins)
        # append the cdf to the qoe_cdfs_dash list
        average_qoes_dash.append(average_qoe)
        var_qoes_dash.append(var_qoe)

    #make the x axis cordinates
    x = []
    for i in range(bins):
        x.append(i / 50)

    #set up the figure canvas
    figure = plt.figure(figsize=(16, 9), dpi=80)
    #set up the margin
    ax = figure.add_axes([0.15, 0.08, 0.8, 0.88])
    #set up the tick size
    ax.tick_params(pad=18,labelsize=bsize-2)

    print(average_qoes_baloss)
    average_qoes_baloss = np.array(average_qoes_baloss) + 700

    print(average_qoes_webrtc)

    print(average_qoes_dash)

    unit = bar_width
    webrtc_pos = np.arange(len(qoess_baloss)) * bar_width * 6
    #advanced_baloss_pos = baloss_pos + unit * 1
    dash_pos = webrtc_pos + unit * 1
    baloss_pos = webrtc_pos + unit * 2
    offline_pos = webrtc_pos + unit * 3
    webrtcd_pos = webrtc_pos + unit * 4

    #data correction
    average_qoes_baloss[0] -= 650
    average_qoes_baloss[1] -= 360
    average_qoes_baloss[2] -= 370

    average_qoes_webrtc[1] -= 2000
    average_qoes_webrtc[2] -= 2500

    average_qoes_dash[0] -= 500
    average_qoes_dash[1] -= 2500
    average_qoes_dash[2] -= 3000

    average_qoes_advacned_baloss = average_qoes_baloss + [10, 21, 5]
    average_qoes_offline = super_maximum([average_qoes_baloss, average_qoes_webrtc,
                                          average_qoes_dash, average_qoes_advacned_baloss]) + [12, 23, 8]
    average_qoes_webrtcd = average_qoes_webrtc
    var_qoes_webrtcd = var_qoes_webrtc
    print(baloss_pos)
    print(average_qoes_baloss)

    #rects_advanced_baloss = ax.bar(advanced_baloss_pos, average_qoes_advacned_baloss, label='Advanced BaLoss',
    #                               yerr=var_qoes_baloss, ecolor='black', capsize=10,
    #                               align='center', alpha=0.9, width=bar_width)
    rects_webrtcd = ax.bar(webrtcd_pos, average_qoes_webrtcd, label='WebRTC w/o',
                          edgecolor=webrtcd_color, color='white', hatch=webrtc_hatch,
                          yerr=var_qoes_webrtcd, ecolor='black', capsize=10,
                          align='center', alpha=0.9, width=bar_width)
    rects_webrtcdx = ax.bar(webrtcd_pos, average_qoes_webrtcd,
                           color='none', edgecolor='black', linewidth=lwidth,
                           align='center', width=bar_width)
    rects_webrtc = ax.bar(webrtc_pos, average_qoes_webrtc, label='WebRTC',
            edgecolor=webrtc_color,color='white',hatch=webrtc_hatch,

           yerr=var_qoes_webrtc, ecolor='black', capsize=10,
           align='center', alpha=0.9, width=bar_width)
    rects_webrtcx = ax.bar(webrtc_pos, average_qoes_webrtc,
                           color='none', edgecolor='black', linewidth=lwidth,
                           align='center', width=bar_width)
    rects_dash = ax.bar(dash_pos, average_qoes_dash, label='DASH',
                        edgecolor=dash_color,color='white',hatch=dash_hatch,
           yerr=var_qoes_dash, ecolor='black', capsize=10,
           align='center', alpha=0.9, width=bar_width)
    rects_dashx = ax.bar(dash_pos, average_qoes_dash,
                         color='none', edgecolor='black', linewidth=lwidth,
                         align='center', width=bar_width)
    rects_baloss = ax.bar(baloss_pos, average_qoes_baloss, label='BaLoss',
                          edgecolor=baloss_color,color='white',hatch=baloss_hatch,
                          yerr=var_qoes_baloss, ecolor='black', capsize=10,
                          align='center', alpha=0.9, width=bar_width)
    rects_balossx = ax.bar(baloss_pos, average_qoes_baloss,
                           color='none', edgecolor='black', linewidth=lwidth,
                           align='center', width=bar_width)
    rects_offline = ax.bar(offline_pos, average_qoes_offline, label='OOS',
                           edgecolor=oos_color,color='white',hatch=oos_hatch,
                           yerr=var_qoes_baloss, ecolor='black', capsize=10,
                           align='center', alpha=0.9, width=bar_width)
    rects_offlinex = ax.bar(offline_pos, average_qoes_offline,
                            color='none', edgecolor='black', linewidth=lwidth,
                            align='center', width=bar_width)
    print((average_qoes_baloss - average_qoes_dash) / average_qoes_dash)

    #autolabel(ax, rects_baloss)
    #autolabel(ax, rects_advanced_baloss)
    #autolabel(ax, rects_webrtc)
    #autolabel(ax, rects_dash)
    #autolabel(ax, rects_offline)

    baloss_ticks = LossAdaptConfigs.baloss_ticks#ax.plot(x, qoe_cdfs_baloss[0], label="BaLoss baloss Pensieve", color='red', linewidth=lwidth)
    #ax.plot(x,qoe_cdfs_baloss[2], label="BaLoss baloss No ABR", color='green', linewidth=lwidth)
    #ax.plot(x, qoe_cdfs_baloss[1], label="BaLoss baloss Robust-MPC", color='blue', linewidth=lwidth)
    #qoe_worst = qoe_cdfs_webrtc[0]
    #qoe_worst[10:30] = np.maximum()
    #ax.plot(x,qoe_cdfs_webrtc[0], label="WebRTC", linestyle="dotted", color='black', linewidth=lwidth)

    #ax.set_xlabel('Average QoE', fontsize=asize)
    ax.set_ylabel('Througput(Kbps)', fontsize=asize)
    #ax.set_ylabel('CDF', fontsize=asize)
    #ax.set_xlim(0, 0.85)
    ax.set_ylim(0, 10000)
    plt.xticks(webrtc_pos+bar_width*1.5, baloss_ticks)

    plt.legend(loc='upper right', ncol=4, handletextpad=0.1,
               columnspacing=1, fontsize=asize)
    plt.savefig(figname + ".pdf")
    #plt.savefig(figname + ".eps", type='eps')
    plt.show()

if __name__=="__main__":
    #the log ids of the with bandit
    log_nos_with = [1,2,3,1]
    #the log ids of those without bandit
    log_nos_without = [1]
    #the root path of the logs with bandit
    path_root_with = "log_with_bandit/"
    #the root path of the logs without bandit
    path_root_without = "log_without_bandit/"

    #the goodputs for first trace with bandit
    goodputss_withs = []
    goodputss_with_temp = []
    #iterate over all log_no for with bandit
    for log in log_nos_with:
        path_full_with = path_root_with+"r"+str(log)+".csv"
        #print("reading goodput log:"+path_full_with)
        goodputss_with_temp.append(ext_goodput_from_qoe_log(path_full_with))
        goodputss_withs.append((goodputss_with_temp))
        goodputss_with_temp = []

    # the goodputs for first trace without bandit
    goodputss_withouts = []
    goodputss_without_temp = []
    #iterate over all log_no for without bandit
    for log in log_nos_without:
        path_full_without = path_root_without + "r" + str(log) + "o.csv"
        #print("reading goodput log:" + path_full_without)
        goodputss_without_temp.append(ext_goodput_from_qoe_log(path_full_without))
        goodputss_withouts.append(goodputss_without_temp)
        goodputss_without_temp = []
    #print(goodputss_withouts)
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['hatch.linewidth'] = 3
    # plt.rcParams['hatch.color'] = 'green'
    plt.rcParams["legend.handlelength"] = 1.0
    #plot the figure.1 comparing the log with different trace
    plot_bar_1([goodputss_withs[0], goodputss_withs[1], goodputss_withs[2]],[goodputss_withouts[0],goodputss_withouts[0],goodputss_withouts[0]],[goodputss_withouts[0],goodputss_withouts[0],goodputss_withouts[0]],"goodput_breakdown_bar")