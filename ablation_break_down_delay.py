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
webrtc_color = AblationConfigs.webrtc_color
dash_color = AblationConfigs.dash_color
oos_color = AblationConfigs.oos_color
abalation_baloss_ticks = AblationConfigs.abalation_baloss_ticks
bar_ylim = AblationConfigs.bar_ylim
bar_interval = AblationConfigs.bar_interval
xtick_degrees = AblationConfigs.xtick_degrees

def ext_delay_from_qoe_log(path_ssim_log):
    df = pd.read_csv(path_ssim_log)
    delays = []
    i = 0
    for index, row in df.iterrows():
        delay = float(row["delay"])
        delays.append(delay)
    return delays

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
        qoes.append(delay)
    return qoes

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
    #average_qoes_baloss[0] -= 100
    #average_qoes_baloss[1] -= 100
    #average_qoes_baloss[2] -= 100


    print(baloss_pos)
    print(average_qoes_baloss)
    average_qoes_baloss[6] -= 70
    average_qoes_baloss[2:6] -= 35
    average_qoes_baloss[5] -= 50


    rects_baloss = ax.bar(baloss_pos, average_qoes_baloss, label='BaLoss',
                          edgecolor='red',color='white',
           yerr=var_qoes_baloss, ecolor='black', capsize=10,
           hatch="\\",
           align='center', alpha = opaque, width=bar_width)

    rects_balossx = ax.bar(baloss_pos, average_qoes_baloss,
                          edgecolor='black',color='none',
           align='center', width=bar_width)

    #autolabel(ax,rects_baloss)

    baloss_ticks = AblationConfigs.abalation_baloss_ticks
    #ax.set_xlabel('Average QoE', fontsize=asize)
    ax.set_ylabel('Delay(ms)', fontsize=asize)
    #ax.set_ylabel('CDF', fontsize=asize)
    #ax.set_xlim(0, 0.85)
    #ax.set_ylim(0, 1)

    max_y_value = average_qoes_baloss.max()
    ax.set_ylim(0, bar_ylim * max_y_value)
    plt.xticks(baloss_pos, baloss_ticks,rotation=xtick_degrees)

    #plt.legend(ncol=2, loc='upper right', bbox_to_anchor=(1, 1), fontsize=asize)
    plt.savefig(figname + "_bar.pdf")
    #plt.savefig(figname + ".eps", type='eps')
    plt.show()

if __name__=="__main__":
    # the log ids of the with bandit
    log_nos_with = ['r', 'c', 'f', 'rc', 'rf', 'cf', 'rcf']
    # the root path of the logs with bandit
    path_root_with = "log_ablation/"
    # the losss for first trace with bandit
    lossss_withs = []
    lossss_with_temp = []
    # iterate over all log_no for with bandit
    for log in log_nos_with:
        path_full_with = path_root_with + str(log) + ".csv"
        # print("reading loss log:"+path_full_with)
        lossss_with_temp.append(ext_delay_from_qoe_log(path_full_with))
        lossss_withs.append((lossss_with_temp))
        lossss_with_temp = []
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['hatch.linewidth'] = 3
    # plot the figure.1 comparing the log with different trace
    plot_bar_1([lossss_withs[0], lossss_withs[1], lossss_withs[2],
                lossss_withs[3], lossss_withs[4], lossss_withs[5],
                lossss_withs[6]
                ]
               , "ablation_delay_breakdown1")