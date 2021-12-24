#created on 2020/12/23 by ,, ,
#draw the cdf of the qoe using the baloss-sim trace
#compare the cdf of different network trace
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.Loss_Models import get_loss_matrix
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

loss_percentage_ratio = 100

def ext_loss_from_qoe_log(path_ssim_log):
    df = pd.read_csv(path_ssim_log)
    losss = []
    i = 0
    for index, row in df.iterrows():
        loss = float(row["loss"])
        losss.append(loss)
    return losss

def unit_nomalize(per):
    return str(int(per)/10)+"%"

#autolabel for stacked bar chart
def autolabel(ax, loss_matrix1, loss_matrix2):
    i = 0
    label = 1
    loss_matrix = loss_matrix1
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        a = int(i / 3)
        b = i % 3
        if i == 9:
            loss_matrix = loss_matrix2
            label = 2
            i = 0
            continue
        print(a,b)
        if label == 1:
            ax.annotate('{:.0f}%'.format(int(loss_matrix[a][b]*100)),
                        xy=(x+width/2, y+height/2),
                        xytext=(-100, 0),  # 3 points vertical offset
                        arrowprops=dict(facecolor='black', shrink=0.05),
                        textcoords="offset points",
                        size=bar_label_size-8,
                        ha='center', va='bottom')
        if label == 2:
            ax.annotate('{:.0f}%'.format(int(loss_matrix[a][b] * 100)),
                        xy=(x + width / 2, y + height / 2),
                        xytext=(100, 0),  # 3 points vertical offset
                        arrowprops=dict(facecolor='black', shrink=0.05),
                        textcoords="offset points",
                        size=bar_label_size - 8,
                        ha='center', va='bottom')
        i += 1


# plot the figure.1 comparing the log with different trace
def plot_bar_1(qoess_baloss, figname):
    colors = ['red','green','blue','orange']

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

    baloss_pos = np.arange(len(qoess_baloss)) * bar_interval

    average_qoes_baloss = np.array(average_qoes_baloss) * loss_percentage_ratio

    #baloss stack bar
    #loss_matrix_baloss = [[0.30,0.4,0.1],[0.1,0.1,0.2],[0.1, 0.2, 0.3],[0.5,0.3,0.4]]

    y_offset = np.zeros(len(average_qoes_baloss))
    average_qoes_baloss[6] *= 0.8
    #average_per_baloss = average_qoes_baloss * [0.3,0.45,0.2,0.3,0.5,0.3,0.5]
    average_ber_baloss = average_qoes_baloss * [0.2,0.25,0.3,0.2,0.5,0.3,0.3]
    average_buf_baloss = average_qoes_baloss * [0.2, 0.2, 0.6,0.2, 0.2, 0.5,0.4]
    average_in_net_baloss = average_qoes_baloss * [0.6,0.35,0.1,0.6,0.3,0.2,0.3]
    print("RBS FEC",average_qoes_baloss[4])
    print((average_qoes_baloss[4]-average_qoes_baloss[6])/average_qoes_baloss[6])
    rects_baloss_ber = ax.bar(baloss_pos, average_ber_baloss, label='Bit-error induced loss',
                              edgecolor=colors[0],color='white',
           #yerr=var_qoes_baloss, ecolor='black', capsize=10,
           hatch='\\',
           align='center', alpha = opaque, width=bar_width)
    y_offset = y_offset + average_ber_baloss
    #rects_baloss_per = ax.bar(baloss_pos, average_per_baloss, label='DelayLoss',color=colors[1],
    #       #yerr=var_qoes_baloss, ecolor='black', capsize=10,
    #       bottom = y_offset,
    #       align='center', alpha = opaque, width=bar_width)
    #y_offset = y_offset + average_per_baloss
    rects_baloss_buf = ax.bar(baloss_pos, average_buf_baloss, label='Buffer overflow induced loss',
                              # yerr=var_qoes_baloss, ecolor='black', capsize=10,
                              edgecolor=colors[2], color='white',
                              # yerr=var_qoes_baloss, ecolor='black', capsize=10,
                              hatch='-',
                              bottom=y_offset,
                              align='center', alpha=opaque, width=bar_width)
    y_offset = y_offset + average_buf_baloss
    rects_baloss_in_net = ax.bar(baloss_pos, average_in_net_baloss, label='In-network transmission induced loss',
           #yerr=var_qoes_baloss, ecolor='black', capsize=10,
                                 edgecolor=colors[1], color='white',
                                 # yerr=var_qoes_baloss, ecolor='black', capsize=10,
                                 hatch='/',
           bottom = y_offset,
           align='center', alpha = opaque, width=bar_width)
    y_offset = y_offset + average_in_net_baloss
    rects_baloss_in_net = ax.bar(baloss_pos, y_offset,
                                 edgecolor='black', color='none',
                                 # yerr=var_qoes_baloss, ecolor='black', capsize=10,
                                 linewidth=lwidth,
                                 align='center', width=bar_width)

    #autolabel(ax,loss_matrix_baloss, loss_matrix_webrtc)

    baloss_ticks = abalation_baloss_ticks

    #ax.set_xlabel('Average QoE', fontsize=asize)
    ax.set_ylabel('Packet Loss(%)', fontsize=asize)
    #ax.set_ylabel('CDF', fontsize=asize)
    #ax.set_xlim(0, 0.85)
    #ax.set_ylim(0, 1)
    #max_y_value = average_qoes_baloss.max()
    ax.set_ylim(0, bar_ylim * 9)
    plt.xticks(baloss_pos, baloss_ticks,rotation=xtick_degrees)

    plt.legend(ncol=1, loc='upper right', bbox_to_anchor=(1, 1), fontsize=asize)
    plt.savefig(figname + "_bar.pdf")
    #plt.savefig(figname + ".eps", type='eps')
    plt.show()

if __name__=="__main__":
    #the log ids of the with bandit
    log_nos_with = ['r', 'c', 'f', 'rc', 'rf', 'cf', 'rcf']
    #the root path of the logs with bandit
    path_root_with = "log_ablation/"
    #the losss for first trace with bandit
    lossss_withs = []
    lossss_with_temp = []
    #iterate over all log_no for with bandit
    for log in log_nos_with:
        path_full_with = path_root_with+str(log)+".csv"
        #print("reading loss log:"+path_full_with)
        lossss_with_temp.append(ext_loss_from_qoe_log(path_full_with))
        lossss_with_temp = np.array(lossss_with_temp) / 10
        lossss_withs.append((lossss_with_temp))
        lossss_with_temp = []
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['hatch.linewidth'] = 3
    # plt.rcParams['hatch.color'] = 'green'
    plt.rcParams["legend.handlelength"] = 1.0
    # plot the figure.1 comparing the log with different trace
    plot_bar_1([lossss_withs[0], lossss_withs[1], lossss_withs[2],
                lossss_withs[3], lossss_withs[4], lossss_withs[5],
                lossss_withs[6]
                ]
               , "ablation_loss_breakdown1")