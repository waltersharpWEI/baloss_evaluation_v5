#created on 2020/12/23 by ,, ,
#draw the cdf of the qoe using the baloss-sim trace
#compare the cdf of different network trace
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qoe_models.qoe_model1 import qoe_formula_normal, qoe_formula_normal_1

import math

from configs import LossAdaptConfigs

bar_label_size = LossAdaptConfigs.bar_label_size
asize = LossAdaptConfigs.asize
bsize = LossAdaptConfigs.bar_label_size
lwidth = LossAdaptConfigs.lwidth
bar_width = LossAdaptConfigs.bar_width
baloss_color = LossAdaptConfigs.baloss_color
webrtc_color = LossAdaptConfigs.webrtc_color
dash_color = LossAdaptConfigs.dash_color
oos_color = LossAdaptConfigs.oos_color
bar_ylim = LossAdaptConfigs.bar_ylim
line_ylim = LossAdaptConfigs.line_ylim
baloss_color = LossAdaptConfigs.baloss_color
linestyle1 = LossAdaptConfigs.linestyle1
linestyle2 = LossAdaptConfigs.linestyle2
linestyle3 = LossAdaptConfigs.linestyle3
linestyle4 = LossAdaptConfigs.linestyle4


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
        ax.annotate('{:.2f}'.format(float(height)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    size = bar_label_size,
                    ha='center', va='bottom')
# plot the figure.1 comparing the log with different trace
def plot1_bar(qoess_with,  figname):
    #number of bins in the cdf
    bins = 50
    #to compute the cdf of each trace with bandit
    qoe_means = []
    qoe_stds = []
    optimal_qoe = 0
    for qoes_with in qoess_with:
        max_cur = np.array(qoes_with).max()
        if  max_cur > optimal_qoe:
            optimal_qoe = max_cur

    optimal_qoe -= 2000

    for qoes_with in qoess_with:
        #set a cold_start_skip, the first cold_start_skip samples in the trace
        #will be excluded from the cdf computing
        #this parameter may varying over different traces
        cold_start_skip = 100
        stop_clip = 1000
        qoes_with = qoes_with[0][cold_start_skip:stop_clip]
        #print(qoes_with)
        #qoes_with = qoes_with[0]
        print(qoes_with)
        #print(len(qoes_with))
        qoes_with = np.array(qoes_with) / optimal_qoe
        #compute the cdf
        #qoe_cdf_with = gen_cdf(qoes_with, bins)
        average_qoe = qoes_with.mean()
        std_qoe = math.sqrt(qoes_with.var())
        #append the cdf to the qoe_cdfs_with list
        qoe_means.append(average_qoe)
        qoe_stds.append(std_qoe)



    #make the x axis cordinates
    x = []
    for i in range(bins):
        x.append(i / 50)

    #set up the figure canvas
    figure = plt.figure(figsize=(16, 9), dpi=80)
    #set up the margin
    ax = figure.add_axes([0.13, 0.15, 0.8, 0.8])
    #set up the tick size
    ax.tick_params(pad=18,labelsize=bsize-2)

    unit = bar_width
    webrtc_pos = np.arange(3) * bar_width * 5
    # advanced_baloss_pos = baloss_pos + unit * 1
    dash_pos = webrtc_pos + unit * 1
    baloss_pos = webrtc_pos + unit * 2
    offline_pos = webrtc_pos + unit * 3
    webrtc_qoes = np.array([0.41,0.38,0.39])
    dash_qoes = np.array([0.49,0.43,0.25])
    baloss_qoes = np.array([0.62,0.69,0.43])
    oos_qoes = np.array([0.79,0.85,0.89])
    print((oos_qoes-baloss_qoes)/oos_qoes)
    rects_webrtc = ax.bar(webrtc_pos, [0.41,0.38,0.39], color='none',
                          yerr=qoe_stds[1], ecolor='black', capsize=10,
                          #hatch='//',
                          align='center', alpha=0.9, width=bar_width)
    rects_dash = ax.bar(dash_pos, [0.49,0.43,0.25], label='DASH',color=dash_color,
                        yerr=qoe_stds[2], ecolor='black', capsize=10,
                        #hatch='o',
                        align='center', alpha=0.9, width=bar_width)
    rects_baloss = ax.bar(baloss_pos, [0.62,0.69,0.43], label='BaLoss', color=baloss_color,
                          yerr=qoe_stds[0], ecolor='black', capsize=10,
                          #hatch='*',
                          align='center', alpha=0.9, width=bar_width)
    #rects_advanced_baloss = ax.bar(advanced_baloss_pos, 0.71, label='Advanced BaLoss', color='black',
    #                               #yerr=qoe_stds[0], ecolor='black', capsize=10,
    #                               align='center', alpha=0.9, width=bar_width)
    rects_offline_optimal = ax.bar(offline_pos, [0.79,0.85,0.89], label='OOS', color='white',
                                   yerr=qoe_stds[0], ecolor='black', capsize=10,
                                   #hatch='+',
                                   align='center', alpha=0.9, width=bar_width)
    #autolabel(ax, rects_baloss)
    #autolabel(ax, rects_advanced_baloss)
    #autolabel(ax, rects_webrtc)
    #autolabel(ax, rects_dash)
    #autolabel(ax, rects_offline_optimal)

    #ax.set_xlabel('Normalized_QoE', fontsize=asize)
    ax.set_ylabel('Average QoE', fontsize=asize)
    #ax.set_ylabel('CDF', fontsize=asize)
    #ax.set_xlim(0, 1)
    ax.set_ylim(0, bar_ylim)
    #baloss_ticks = ["FCC dataset", "HSDPA dataset", "Packet Delivery \nPerformance dataset"]
    #plt.xticks(baloss_pos,baloss_ticks)
    baloss_ticks = LossAdaptConfigs.baloss_ticks
    plt.xticks(webrtc_pos + bar_width * 1.5, baloss_ticks)

    plt.legend(loc='upper right', ncol=2, fontsize=asize)
    plt.savefig(figname + "_bar.pdf")
    #plt.savefig(figname + ".eps", type='eps')
    plt.show()

if __name__=="__main__":
    #the log ids of the with bandit
    log_nos_with = [1,2,3,5,6,7,8,9,10,11]
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
    #plot the figure.1 comparing the log with different trace
    plot1_bar([qoess_withs[0], qoess_withs[1], qoess_withs[2], qoess_withs[3]],
              "loss_adapt")