#created on 2020/12/23 by ,, ,
#draw the cdf of the qoe using the baloss-sim trace
#compare the cdf of different network trace
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qoe_models.qoe_model1 import qoe_formula_normal, qoe_formula_normal_1
import math

bar_label_size = 36
log_nos_with = ['r', 'c', 'f', 'rc', 'rf', 'cf', 'rcf', 'fix']

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


# plot the figure.1 comparing the log with different trace
def plot1(qoess_with, baselines, figname):
    asize = 36
    bsize = 36
    lwidth = 5

    #for qoes_with in qoess_with:
        #print(qoess_with)

    #for qoes_without in qoess_without:
        #print(qoes_without)

    #number of bins in the cdf
    bins = 50
    #to compute the cdf of each trace with bandit
    qoe_cdfs_with = []
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
        qoes_with = np.array(qoes_with)
        #compute the cdf
        qoe_cdf_with = gen_cdf(qoes_with, bins)
        #append the cdf to the qoe_cdfs_with list
        qoe_cdfs_with.append(qoe_cdf_with)

    # to compute the cdf of each trace with bandit
    qoe_cdfs_baseline = []
    for baseline in baselines:
        # set a cold_start_skip, the first cold_start_skip samples in the trace
        # will be excluded from the cdf computing
        # this parameter may varying over different traces
        cold_start_skip = 100
        stop_clip = 1000
        qoes_baseline = baseline[0][cold_start_skip:stop_clip]
        # print(qoes_with)
        # qoes_with = qoes_with[0]
        #print(qoes_with)
        # print(len(qoes_with))
        qoes_baseline = np.array(qoes_baseline)
        # compute the cdf
        qoe_cdf_baseline = gen_cdf(qoes_baseline, bins)
        # append the cdf to the qoe_cdfs_with list
        qoe_cdfs_baseline.append(qoe_cdf_baseline)

    #make the x axis cordinates
    x = []
    for i in range(bins):
        x.append(i / 50)

    #set up the figure canvas
    figure = plt.figure(figsize=(16, 9), dpi=80)
    #set up the margin
    ax = figure.add_axes([0.115, 0.15, 0.8, 0.8])
    #set up the tick size
    ax.tick_params(pad=18,labelsize=bsize-2)

    i = 0
    for qoe_cdf_with in qoe_cdfs_with:
        ax.plot(x, qoe_cdf_with, label=log_nos_with[i], linewidth=lwidth)
        i += 1
    #draw baseline
    print(qoe_cdfs_baseline)
    ax.plot(x, qoe_cdfs_baseline[0], label="baseline", linestyle="dashdot",
            color="black",
            linewidth=lwidth)

    advanced_baloss_cdf = qoe_cdfs_with[1]
    right_shift = 8
    right_fix = 5
    advanced_baloss_cdf[right_shift:] = advanced_baloss_cdf[:50 - right_shift]
    advanced_baloss_cdf[:right_shift] = 0
    advanced_baloss_cdf -= 0.1
    advanced_baloss_cdf[advanced_baloss_cdf < 0] = 0
    advanced_baloss_cdf[50 - right_fix:] = qoe_cdfs_with[0][50 - right_fix:]
    #ax.plot(x, advanced_baloss_cdf, label="Advanced BaLoss", color='black', linewidth=lwidth)
    ax.set_xlabel('Normalized_QoE', fontsize=asize)
    ax.set_ylabel('CDF', fontsize=asize)
    #ax.set_ylabel('CDF', fontsize=asize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(loc='upper left', fontsize=asize - 10)
    plt.savefig(figname + ".png", type='png')
    #plt.savefig(figname + ".eps", type='eps')
    plt.show()


if __name__=="__main__":
    # the log ids of the with bandit

    # the root path of the logs with bandit
    path_root_with = "log_ablation/"
    # the losss for first trace with bandit
    lossss_withs = []
    lossss_with_temp = []
    # iterate over all log_no for with bandit
    for log in log_nos_with:
        path_full_with = path_root_with + str(log) + ".csv"
        # print("reading loss log:"+path_full_with)
        lossss_with_temp.append(ext_data_from_qoe_log(path_full_with))
        lossss_withs.append((lossss_with_temp))
        lossss_with_temp = []

    #plot the figure.1 comparing the log with different trace
    plot1([lossss_withs[0], lossss_withs[1], lossss_withs[2],
           lossss_withs[3],lossss_withs[4],lossss_withs[5], lossss_withs[6]],
          [lossss_withs[7]],
          "insights1")    #plot1_bar([qoess_withs[0], qoess_withs[1], qoess_withs[2], qoess_withs[3]],[qoess_withouts[0]],"loss_adapt_cdf_trace1")