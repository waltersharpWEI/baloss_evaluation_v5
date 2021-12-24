#created on 2020/12/23 by ,, ,
#draw the cdf of the qoe using the baloss-sim trace
#updated
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

    for qoes_with in qoess_with:
        print(qoess_with)

    for qoes_without in qoess_without:
        print(qoess_without)

    bins = 50
    ssimss_cdf = []
    qoe_cdfs_with = []
    cold_start_skip = 500
    for qoes_with in qoess_with:
        qoes_with = qoes_with[0][cold_start_skip:]
        #print(len(qoes_with))
        qoes_with = np.array(qoes_with)
        #s = gen_pdf(ssims, bins)
        qoe_cdf_with = gen_cdf(qoes_with, bins)
        qoe_cdfs_with.append(qoe_cdf_with)

    for qoes_with in qoess_with:
        qoes_with = qoes_with[0][cold_start_skip:]
        # print(len(qoes_with))
        qoes_with = np.array(qoes_with)
        # s = gen_pdf(ssims, bins)
        s = gen_cdf(qoes_with, bins)
        ss.append(s)

    #print(ss)
    x = []
    for i in range(bins):
        x.append(i / 50)

    figure = plt.figure(figsize=(16, 9), dpi=80)
    ax = figure.add_axes([0.115, 0.15, 0.8, 0.8])
    ax.tick_params(pad=18,labelsize=bsize-2)



    ax.plot(ss[0], label="Bandit", color='red', linewidth=lwidth)
    ax.plot(ss[1], label="No Bandit(average)", color='blue', linewidth=lwidth)

    ax.set_xlabel('QoE', fontsize=asize)
    ax.set_ylabel('CDF', fontsize=asize)
    #ax.set_ylabel('CDF', fontsize=asize)
    #ax.set_xlim(0, 1)
    #ax.set_ylim(0, 1)
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
    qoess_with_a = []
    # the qoes for second trace with bandit
    qoess_with_b = []
    # the qoes for third trace with bandit
    qoess_with_c = []
    # the qoes for first trace without bandit
    qoess_without = []

    #iterate over all log_no for with bandit
    for log in log_nos_with:
        path_full_with = path_root_with+"r"+str(log)+".csv"
        print("reading qoe log:"+path_full_with)
        qoess_with_a.append(ext_data_from_qoe_log(path_full_with))

    #iterate over all log_no for without bandit
    for log in log_nos_without:
        path_full_without = path_root_without + "r" + str(log) + "o.csv"
        print("reading qoe log:" + path_full_without)
        qoess_without.append(ext_data_from_qoe_log(path_full_without))
    #plot the figure.1 comparing the log with different trace
    plot1([qoess_with_a, qoess_with_b, qoess_with_c],[qoess_without],"fig3c")