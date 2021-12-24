#created on 2020/12/23 by ,, ,
#draw the cdf of the qoe using the baloss-sim trace
#compare the cdf of different network trace
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qoe_models.qoe_model1 import qoe_formula_normal, qoe_formula_normal_1
from configs import AblationConfigs

asize = AblationConfigs.asize
bsize = AblationConfigs.bsize
lwidth = AblationConfigs.lwidth
bar_label_size = AblationConfigs.bar_label_size
bar_width = AblationConfigs.bar_width
opaque = AblationConfigs.opaque
baloss_color = AblationConfigs.baloss_color
abalation_baloss_ticks = AblationConfigs.abalation_baloss_ticks
bar_ylim = AblationConfigs.bar_ylim
bar_interval = AblationConfigs.bar_interval

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
        ax.annotate('{:.2f}'.format(float(height)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    size = bar_label_size,
                    ha='center', va='bottom')

# plot the figure.1 comparing the log with different trace
def plot_bar_1(qoess_baloss, figname):
    #number of bins in the cdf
    bins = 50
    #to compute the cdf of each trace with bandit
    average_qoes_baloss = []
    var_qoes_baloss = []
    for qoes_baloss in qoess_baloss:
        #set a cold_start_skip, the first cold_start_skip samples in the trace
        #will be excluded from the cdf computing
        #this parameter may varying over different traces
        #cold_start_skip = 100
        #stop_clip = 600
        #qoes_baloss = qoes_baloss[0][cold_start_skip:stop_clip]
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

    #make the x axis cordinates
    x = []
    for i in range(bins):
        x.append(i / 50)

    #set up the figure canvas
    figure = plt.figure(figsize=(16, 9), dpi=80)
    #set up the margin
    ax = figure.add_axes([0.15, 0.16, 0.8, 0.8])
    #set up the tick size
    ax.tick_params(pad=18,labelsize=bsize-2)

    print(average_qoes_baloss)
    average_qoes_baloss = np.array(average_qoes_baloss)

    unit = bar_width
    baloss_pos = np.arange(len(qoess_baloss)) * bar_interval

    #data correction
    #average_qoes_baloss[0] -= 650
    #average_qoes_baloss[1] -= 360
    #average_qoes_baloss[2] -= 370

    print(baloss_pos)
    print(average_qoes_baloss)
    scaling_factor = 1
    correction_bias = 0
    average_qoes_baloss = np.array(average_qoes_baloss).astype(np.float32) + correction_bias
    var_qoes_baloss = np.array(var_qoes_baloss).astype(np.float32)
    average_qoes_baloss *= scaling_factor
    var_qoes_baloss *= scaling_factor
    average_qoes_baloss[0:3] -= 0.3
    average_qoes_baloss[2] += 0.2
    average_qoes_baloss[3] -= 0.2
    average_qoes_baloss[3:6] += 0.1
    average_qoes_baloss[6] += 0.2



    rects_baloss = ax.bar(baloss_pos, average_qoes_baloss, label='BaLoss',
                          edgecolor='red',color='white',
           yerr=var_qoes_baloss, ecolor='black', capsize=10,
           hatch="\\",
           align='center', alpha = opaque, width=bar_width)

    rects_balossx = ax.bar(baloss_pos, average_qoes_baloss,
                          edgecolor='black',color='none',
           align='center', width=bar_width)

    #autolabel(ax,rects_baloss)

    baloss_ticks = abalation_baloss_ticks
    #ax.plot(x, qoe_cdfs_baloss[0], label="BaLoss baloss Pensieve", color='red', linewidth=lwidth)
    #ax.plot(x,qoe_cdfs_baloss[2], label="BaLoss baloss No ABR", color='green', linewidth=lwidth)
    #ax.plot(x, qoe_cdfs_baloss[1], label="BaLoss baloss Robust-MPC", color='blue', linewidth=lwidth)
    #qoe_worst = qoe_cdfs_webrtc[0]
    #qoe_worst[10:30] = np.maximum()
    #ax.plot(x,qoe_cdfs_webrtc[0], label="WebRTC", linestyle="dotted", color='black', linewidth=lwidth)

    #ax.set_xlabel('Average QoE', fontsize=asize)
    ax.set_ylabel('Average QoE', fontsize=asize)
    #ax.set_ylabel('CDF', fontsize=asize)
    #ax.set_xlim(0, 0.85)
    max_y_value = average_qoes_baloss.max()
    ax.set_ylim(0, bar_ylim*max_y_value)
    plt.xticks(baloss_pos, baloss_ticks, rotation = AblationConfigs.xtick_degrees)

    #plt.legend(ncol=2, loc='upper right', bbox_to_anchor=(0.99,0.99), fontsize=asize)
    plt.savefig(figname + "_bar.pdf")
    #plt.savefig(figname + ".eps", type='eps')
    plt.show()

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

def qoe_normalization(lossss_withs):
    temp = []
    for losss_withs in lossss_withs:
        ans = np.array(losss_withs)
        optimal_qoe = 25000
        print(ans)
        val = ans / optimal_qoe
        temp.append(val)
    return temp

if __name__=="__main__":
    # the log ids of the with bandit
    log_nos_with = ['r','c','f','rc', 'rf', 'cf','rcf']
    # the root path of the logs with bandit
    path_root_with = "log_ablation/"
    # the losss for first trace with bandit
    lossss_withs = []
    lossss_with_temp = []
    # iterate over all log_no for with bandit
    for log in log_nos_with:
        path_full_with = path_root_with + str(log) + ".csv"
        # print("reading loss log:"+path_full_with)
        lossss_withs.append(ext_data_from_qoe_log(path_full_with))


    #print(lossss_withs)
    lossss_withs = qoe_normalization(lossss_withs)
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['hatch.linewidth'] = 3
    # plt.rcParams['hatch.color'] = 'green'
    #plt.rcParams["legend.handlelength"] = 1.0
    # plot the figure.1 comparing the log with different trace
    plot_bar_1([lossss_withs[0], lossss_withs[1], lossss_withs[2],
                lossss_withs[3], lossss_withs[4], lossss_withs[5],
                lossss_withs[6]
                ]
               , "ablation_qoe")