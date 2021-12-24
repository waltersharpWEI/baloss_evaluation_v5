#created on 2020/12/23 by ,, ,
#draw the cdf of the qoe using the baloss-sim trace
#compare the cdf of different network trace
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qoe_models.qoe_model1 import qoe_formula_normal, qoe_formula_normal_1


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
def plot1(qoess_with, qoess_without, figname):
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
        stop_clip = 3000
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

    #to compute the cdf of each trace without bandit
    qoe_cdfs_without = []
    for qoes_without in qoess_without:
        # set a cold_start_skip, the first cold_start_skip samples in the trace
        # will be excluded from the cdf computing
        # this parameter may varying over different traces
        cold_start_skip = 100
        stop_clip = 4000
        qoes_without = qoes_without[0][cold_start_skip:stop_clip]
        # print(qoes_with)
        # qoes_with = qoes_with[0]
        print(qoes_without)
        # print(len(qoes_with))
        qoes_without = np.array(qoes_without)
        # compute the cdf
        qoe_cdf_without = gen_cdf(qoes_without, bins)
        # append the cdf to the qoe_cdfs_without list
        qoe_cdfs_without.append(qoe_cdf_without)

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

    ax.plot(x, qoe_cdfs_with[0], label="BaLoss Iteration=500", color='red', linewidth=lwidth)
    ax.plot(x,qoe_cdfs_with[1], label="BaLoss Iteration=1000", color='green', linewidth=lwidth)
    ax.plot(x, qoe_cdfs_with[2], label="BaLoss Iteration=5000", color='blue', linewidth=lwidth)
    qoe_worst = qoe_cdfs_without[0]
    #qoe_worst[10:30] = np.maximum()
    ax.plot(x, qoe_cdfs_without[0], label="Default WebRTC", linestyle="dashdot", color='black', linewidth=lwidth)

    ax.set_xlabel('Normalized QoE', fontsize=asize)
    ax.set_ylabel('CDF', fontsize=asize)
    #ax.set_ylabel('CDF', fontsize=asize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(loc='upper left', fontsize=asize - 10)
    plt.savefig(figname + ".png", type='png')
    #plt.savefig(figname + ".eps", type='eps')
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

    #plot the figure.1 comparing the log with different trace
    plot1([qoess_withs[0], qoess_withs[1], qoess_withs[2]],[qoess_withouts[0]],"detail_impact_iter")