#created on 2020/12/23 by ,, ,
#draw the cdf of the qoe using the baloss-sim trace
#compare the cdf of different network trace
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qoe_models.qoe_model1 import qoe_formula_normal, qoe_formula_normal_1


def gen_cdf_bins(y,bin_count):
    hist, bin_edges = np.histogram(y,bins=bin_count)
    cdf = np.cumsum(hist) / np.sum(hist)
    #print(cdf)
    return cdf, bin_edges

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


def plot_per(pers,figname):
    asize = 36
    bsize = 36
    lwidth = 5

    # number of bins in the cdf
    bins = 50
    # to compute the cdf of each trace with bandit
    cdfs_with = []
    for per in pers:
        # set a cold_start_skip, the first cold_start_skip samples in the trace
        # will be excluded from the cdf computing
        # this parameter may varying over different traces
        cold_start_skip = 10
        stop_clip = 1000
        per = per[cold_start_skip:stop_clip]
        # print(qoes_with)
        # qoes_with = qoes_with[0]
        print(per)
        # print(len(qoes_with))
        per = np.array(per)
        # compute the cdf
        cdf_with = gen_cdf(per, bins)
        # append the cdf to the qoe_cdfs_with list
        cdfs_with.append(cdf_with)

    # make the x axis cordinates
    x = []
    for i in range(bins):
        x.append(i / 50)

    # set up the figure canvas
    figure = plt.figure(figsize=(16, 9), dpi=80)
    # set up the margin
    ax = figure.add_axes([0.115, 0.15, 0.8, 0.8])
    # set up the tick size
    ax.tick_params(pad=18, labelsize=bsize - 2)

    ax.plot(x, cdfs_with[0], label="trace 1", color='red', linewidth=lwidth)
    ax.plot(x, cdfs_with[1], label="trace 2", color='green', linewidth=lwidth)
    ax.plot(x, cdfs_with[2], label="trace 3", color='blue', linewidth=lwidth)

    ax.set_xlabel('In Network Loss(%)', fontsize=asize)
    ax.set_ylabel('CDF', fontsize=asize)
    # ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(loc='upper left', fontsize=asize - 10)
    plt.savefig(figname + ".png", type='png')
    # plt.show()


def plot_in_nets(in_nets,figname):
    asize = 36
    bsize = 36
    lwidth = 5

    # number of bins in the cdf
    bins = 50
    # to compute the cdf of each trace with bandit
    cdfs_with = []
    for in_net in in_nets:
        # set a cold_start_skip, the first cold_start_skip samples in the trace
        # will be excluded from the cdf computing
        # this parameter may varying over different traces
        cold_start_skip = 10
        stop_clip = 1000
        in_net = in_net[cold_start_skip:stop_clip]
        # print(qoes_with)
        # qoes_with = qoes_with[0]
        print(in_net)
        # print(len(qoes_with))
        in_net = np.array(in_net)
        # compute the cdf
        cdf_with = gen_cdf(in_net, bins)
        # append the cdf to the qoe_cdfs_with list
        cdfs_with.append(cdf_with)

    # make the x axis cordinates
    x = []
    for i in range(bins):
        x.append(i / 50)

    # set up the figure canvas
    figure = plt.figure(figsize=(16, 9), dpi=80)
    # set up the margin
    ax = figure.add_axes([0.115, 0.15, 0.8, 0.8])
    # set up the tick size
    ax.tick_params(pad=18, labelsize=bsize - 2)

    ax.plot(x, cdfs_with[0], label="trace 1", color='red', linewidth=lwidth)
    ax.plot(x, cdfs_with[1], label="trace 2", color='green', linewidth=lwidth)
    ax.plot(x, cdfs_with[2], label="trace 3", color='blue', linewidth=lwidth)

    ax.set_xlabel('In Network Loss(%)', fontsize=asize)
    ax.set_ylabel('CDF', fontsize=asize)
    #ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(loc='upper left', fontsize=asize - 10)
    plt.savefig(figname + ".png", type='png')
    # plt.show()


def plot_ber(bers,figname):
    asize = 36
    bsize = 36
    lwidth = 5

    # number of bins in the cdf
    bins = 50
    # to compute the cdf of each trace with bandit
    cdfs_with = []
    for ber in bers:
        # set a cold_start_skip, the first cold_start_skip samples in the trace
        # will be excluded from the cdf computing
        # this parameter may varying over different traces
        cold_start_skip = 10
        stop_clip = 1000
        ber = ber[cold_start_skip:stop_clip]
        # print(qoes_with)
        # qoes_with = qoes_with[0]
        print(ber)
        # print(len(qoes_with))
        ber = np.array(ber)
        # compute the cdf
        cdf_with = gen_cdf(ber, bins)
        # append the cdf to the qoe_cdfs_with list
        cdfs_with.append(cdf_with)

    # make the x axis cordinates
    x = []
    for i in range(bins):
        x.append(i / 50)

    # set up the figure canvas
    figure = plt.figure(figsize=(16, 9), dpi=80)
    # set up the margin
    ax = figure.add_axes([0.115, 0.15, 0.8, 0.8])
    # set up the tick size
    ax.tick_params(pad=18, labelsize=bsize - 2)

    ax.plot(x, cdfs_with[0], label="trace 1", color='red', linewidth=lwidth)
    ax.plot(x, cdfs_with[1], label="trace 2", color='green', linewidth=lwidth)
    ax.plot(x, cdfs_with[2], label="trace 3", color='blue', linewidth=lwidth)

    ax.set_xlabel('Bit Error-induced Loss', fontsize=asize)
    ax.set_ylabel('CDF', fontsize=asize)
    #ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(loc='upper left', fontsize=asize - 10)
    plt.savefig(figname + ".png", type='png')
    #plt.show()


# plot the figure.1 comparing the log with different trace
def plot1(pers,bers,in_nets,figname_prefix):
    plot_per(pers, figname_prefix + "_per")
    plot_ber(bers,figname_prefix+"_ber")
    plot_in_nets(in_nets, figname_prefix + "_in_net")

if __name__=="__main__":
    #the log ids of the with bandit
    trace_nos = [1,2,3]
    path_root_with = "net_traces/"

    #the qoes for first trace with bandit
    pers = []
    bers = []
    in_nets = []

    #iterate over all log_no for with bandit
    for trace_no in trace_nos:
        path_full_with = path_root_with+"trace"+str(trace_no)+".csv"
        #print("reading qoe log:"+path_full_with)
        df = pd.read_csv(path_full_with)
        per = np.array(df["per"])
        ber = np.array(df["ber"])
        in_net = np.array(df["rtt"]) / 100.0
        pers.append(per)
        bers.append(ber)
        in_nets.append(in_net)

    #plot the figure.1 comparing the log with different trace
    plot1(pers,bers,in_nets,"loss_character_1")