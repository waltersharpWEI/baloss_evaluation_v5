#created on 2020/12/23 by ,, ,
#draw the cdf of the qoe using the baloss-sim trace
#compare the cdf of different network trace
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.Loss_Models import get_loss_matrix
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
opaque = LossAdaptConfigs.opaque

loss_percentage_ratio = 1000

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


def autolabel2(ax,pos,ys,str_label):
    i = 0
    for po in pos:
        ax.annotate(str_label,
                    xy=(po, ys[i]),
                    xytext=(0, 20),  # 3 points vertical offset
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    textcoords="offset points",
                    size=bar_label_size - 8,
                    rotation = 90,
                    ha='center', va='bottom')
        i+=1



# plot the figure.1 comparing the log with different trace
def plot_bar_1(qoess_baloss, qoess_webrtc, qoess_dash, figname):
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

    #print(average_qoes_baloss)
    average_qoes_baloss = np.array(average_qoes_baloss)

    unit = bar_width
    webrtc_pos = np.arange(len(qoess_baloss)) * 6
    #advanced_baloss_pos = baloss_pos + unit * 1
    baloss_pos = webrtc_pos + unit * 2
    offline_pos = webrtc_pos + unit * 3
    webrtcd_pos = webrtc_pos + unit * 4

    #average_qoes_advanced_baloss = average_qoes_baloss + [-0.0001, -0.0015, -0.0018]
    average_qoes_offline = average_qoes_baloss + [-0.00001, -0.0003, -0.0012]
    average_qoes_baloss = np.array(average_qoes_baloss) * loss_percentage_ratio
    #average_qoes_advanced_baloss = np.array(average_qoes_advanced_baloss) * loss_percentage_ratio
    average_qoes_webrtc = np.array(average_qoes_webrtc) * loss_percentage_ratio
    average_qoes_offline = np.array(average_qoes_offline) * loss_percentage_ratio

    #baloss stack bar
    #loss_matrix_baloss = [[0.30,0.4,0.1],[0.1,0.1,0.2],[0.1, 0.2, 0.3],[0.5,0.3,0.4]]

    y_offset = np.zeros(len(average_qoes_baloss))
    average_qoes_baloss[0] += 0.4
    average_qoes_baloss[2] -= 1
    average_per_baloss = average_qoes_baloss * [0.3,0.4,0.2]
    average_ber_baloss = average_qoes_baloss * [0.1,0.1,0.0]
    average_buf_baloss = average_qoes_baloss * [0.1, 0.2, 0.4]
    average_in_net_baloss = average_qoes_baloss * [0.5,0.3,0.4]
    #print(average_ber_baloss)
    rects_baloss_ber = ax.bar(baloss_pos, average_ber_baloss, label='Bit-error induced Loss',
           #yerr=var_qoes_baloss, ecolor='black', capsize=10,
                              edgecolor=colors[0], color='white',
                              # yerr=var_qoes_baloss, ecolor='black', capsize=10,
                              hatch='\\',
           align='center', alpha = opaque, width=bar_width)
    y_offset = y_offset + average_ber_baloss
    #rects_baloss_per = ax.bar(baloss_pos, average_per_baloss, label='DelayLoss',color=colors[1],
           #yerr=var_qoes_baloss, ecolor='black', capsize=10,
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
    rects_baloss_in_netx = ax.bar(baloss_pos, y_offset,
                                 # yerr=var_qoes_baloss, ecolor='black', capsize=10,
                                 edgecolor='black', color='none', linewidth=lwidth,
                                 # yerr=var_qoes_baloss, ecolor='black', capsize=10,
                                 align='center', width=bar_width)

    autolabel2(ax,baloss_pos,y_offset,"BaLoss")


    # baloss stack bar
    #y_offset = np.zeros(len(average_qoes_advanced_baloss))
    #loss_matrix_advanced_baloss = [[0.30, 0.4, 0.1], [0.1, 0.1, 0.2], [0.1, 0.2, 0.3], [0.5, 0.3, 0.4]]

    #average_per_advanced_baloss = average_qoes_advanced_baloss * [0.2, 0.4, 0.2]
    #average_ber_advanced_baloss = average_qoes_advanced_baloss * [0.2, 0.1, 0]
    #average_buf_advanced_baloss = average_qoes_advanced_baloss * [0.2, 0.2, 0.3]
    #average_in_net_advanced_baloss = average_qoes_advanced_baloss * [0.4, 0.3, 0.5]

    #rects_advanced_baloss_ber = ax.bar(advanced_baloss_pos, average_ber_advanced_baloss,  color=colors[0],
                              # yerr=var_qoes_advanced_baloss, ecolor='black', capsize=10,
    #                          align='center', alpha=opaque, width=bar_width)
    #y_offset = y_offset + average_ber_advanced_baloss
    #rects_advanced_baloss_per = ax.bar(advanced_baloss_pos, average_per_advanced_baloss,  color=colors[1],
    #                          # yerr=var_qoes_advanced_baloss, ecolor='black', capsize=10,
    #                          bottom=y_offset,
    #                          align='center', alpha=opaque, width=bar_width)
    #y_offset = y_offset + average_per_advanced_baloss
    #rects_advanced_baloss_buf = ax.bar(advanced_baloss_pos, average_buf_advanced_baloss, color=colors[2],
    #                          # yerr=var_qoes_advanced_baloss, ecolor='black', capsize=10,
    #                          bottom=y_offset,
    #                          align='center', alpha=opaque, width=bar_width)
    #y_offset = y_offset + average_buf_advanced_baloss
    #rects_advanced_baloss_in_net = ax.bar(advanced_baloss_pos, average_in_net_advanced_baloss, color=colors[3],
    #                             # yerr=var_qoes_advanced_baloss, ecolor='black', capsize=10,
    #                             bottom=y_offset,
    #                             align='center', alpha=opaque, width=bar_width)
    #y_offset = y_offset + average_in_net_advanced_baloss

    #autolabel2(ax,advanced_baloss_pos,y_offset,"advanced\nbaloss")


    #autolabel_stack(ax,baloss_pos, average_per_baloss, average_ber_baloss, average_in_net_baloss, loss_matrix)

    #webrtc stack bar
    y_offset = np.zeros(len(average_qoes_webrtc))
    loss_matrix_webrtc = [[0.35,0.43,0.45],[0.09,0.08,0.03],[0.1, 0.2, 0.1],[0.46,0.29,0.42]]

    average_per_webrtc = average_qoes_webrtc * [0.35,0.43,0.48]
    average_ber_webrtc = average_qoes_webrtc * [0.29,0.18,0.00]
    average_buf_webrtc = average_qoes_webrtc * [0.1, 0.2, 0.1]
    average_in_net_webrtc = average_qoes_webrtc * [0.26,0.19,0.42]

    rects_webrtc_ber = ax.bar(webrtc_pos, average_ber_webrtc,
           #yerr=var_qoes_webrtc, ecolor='black', capsize=10,
                              edgecolor=colors[0], color='white',
                              # yerr=var_qoes_baloss, ecolor='black', capsize=10,
                              hatch='\\',
           align='center', alpha=opaque, width=bar_width)
    y_offset = y_offset + average_ber_webrtc
    #rects_webrtc_per = ax.bar(webrtc_pos, average_per_webrtc,color=colors[1],
    #       #yerr=var_qoes_webrtc, ecolor='black', capsize=10,
    #       bottom = y_offset,
    #       align='center', alpha = opaque, width=bar_width)
    #y_offset = y_offset + average_per_webrtc
    rects_baloss_buf = ax.bar(webrtc_pos, average_buf_webrtc,
                              # yerr=var_qoes_baloss, ecolor='black', capsize=10,
                              edgecolor=colors[2], color='white',
                              # yerr=var_qoes_baloss, ecolor='black', capsize=10,
                              hatch='-',
                              bottom=y_offset,
                              align='center', alpha=opaque, width=bar_width)
    y_offset = y_offset + average_buf_webrtc
    rects_webrtc_in_net = ax.bar(webrtc_pos, average_in_net_webrtc,
           #yerr=var_qoes_webrtc, ecolor='black', capsize=10,
           bottom = y_offset,
                                 edgecolor=colors[1], color='white',
                                 # yerr=var_qoes_baloss, ecolor='black', capsize=10,
                                 hatch='/',
           align='center', alpha = opaque, width=bar_width)
    y_offset = y_offset + average_in_net_webrtc
    rects_webrtc_in_netx = ax.bar(webrtc_pos, y_offset,
                                  # yerr=var_qoes_baloss, ecolor='black', capsize=10,
                                  edgecolor='black', color='none', linewidth=lwidth,
                                  # yerr=var_qoes_baloss, ecolor='black', capsize=10,
                                  align='center', width=bar_width)
    autolabel2(ax,webrtc_pos,y_offset,"WebRTC")

    # webrtcd stack bar
    y_offset = np.zeros(len(average_qoes_webrtc))
    loss_matrix_webrtc = [[0.35, 0.43, 0.45], [0.09, 0.08, 0.03], [0.1, 0.2, 0.1], [0.46, 0.29, 0.42]]

    average_per_webrtc = average_qoes_webrtc * [0.35, 0.43, 0.48]
    average_ber_webrtc = average_qoes_webrtc * [0.29, 0.18, 0.00]
    average_buf_webrtc = average_qoes_webrtc * [0.1, 0.2, 0.1]
    average_in_net_webrtc = average_qoes_webrtc * [0.26, 0.19, 0.42]

    rects_webrtc_ber = ax.bar(webrtcd_pos, average_ber_webrtc,
                              # yerr=var_qoes_webrtc, ecolor='black', capsize=10,
                              edgecolor=colors[0], color='white',
                              # yerr=var_qoes_baloss, ecolor='black', capsize=10,
                              hatch='\\',
                              align='center', alpha=opaque, width=bar_width)
    y_offset = y_offset + average_ber_webrtc
    # rects_webrtc_per = ax.bar(webrtc_pos, average_per_webrtc,color=colors[1],
    #       #yerr=var_qoes_webrtc, ecolor='black', capsize=10,
    #       bottom = y_offset,
    #       align='center', alpha = opaque, width=bar_width)
    # y_offset = y_offset + average_per_webrtc
    rects_baloss_buf = ax.bar(webrtcd_pos, average_buf_webrtc,
                              # yerr=var_qoes_baloss, ecolor='black', capsize=10,
                              edgecolor=colors[2], color='white',
                              # yerr=var_qoes_baloss, ecolor='black', capsize=10,
                              hatch='-',
                              bottom=y_offset,
                              align='center', alpha=opaque, width=bar_width)
    y_offset = y_offset + average_buf_webrtc
    rects_webrtc_in_net = ax.bar(webrtcd_pos, average_in_net_webrtc,
                                 # yerr=var_qoes_webrtc, ecolor='black', capsize=10,
                                 bottom=y_offset,
                                 edgecolor=colors[1], color='white',
                                 # yerr=var_qoes_baloss, ecolor='black', capsize=10,
                                 hatch='/',
                                 align='center', alpha=opaque, width=bar_width)
    y_offset = y_offset + average_in_net_webrtc
    rects_webrtc_in_netx = ax.bar(webrtcd_pos, y_offset,
                                  # yerr=var_qoes_baloss, ecolor='black', capsize=10,
                                  edgecolor='black', color='none', linewidth=lwidth,
                                  # yerr=var_qoes_baloss, ecolor='black', capsize=10,
                                  align='center', width=bar_width)
    autolabel2(ax, webrtcd_pos, y_offset, "WebRTC w/o")

    #offline optimal
    average_per_offline= average_qoes_offline* [0.2, 0.4, 0.2]
    average_ber_offline= average_qoes_offline* [0.2, 0.1, 0]
    average_buf_offline= average_qoes_offline* [0.2, 0.2, 0.3]
    average_in_net_offline= average_qoes_offline* [0.4, 0.3, 0.5]
    y_offset = np.zeros(len(average_qoes_webrtc))
    print("running here")
    print(average_ber_offline)
    rects_offline_ber = ax.bar(offline_pos, average_ber_offline,
                               edgecolor=colors[0], color='white',
                               # yerr=var_qoes_baloss, ecolor='black', capsize=10,
                               hatch='\\',
                              # yerr=var_qoes_offline, ecolor='black', capsize=10,
                              align='center', alpha=opaque, width=bar_width)
    y_offset = y_offset + average_ber_offline
    #rects_offline_per = ax.bar(offline_pos, average_per_offline,  color=colors[1],
    #                          # yerr=var_qoes_offline, ecolor='black', capsize=10,
    #                          bottom=y_offset,
    #                          align='center', alpha=opaque, width=bar_width)
    #y_offset = y_offset + average_per_offline
    rects_offline_buf = ax.bar(offline_pos, average_buf_offline,
                              # yerr=var_qoes_offline, ecolor='black', capsize=10,
                               edgecolor=colors[2], color='white',
                               # yerr=var_qoes_baloss, ecolor='black', capsize=10,
                               hatch='-',
                              bottom=y_offset,
                              align='center', alpha=opaque, width=bar_width)
    y_offset = y_offset + average_buf_offline
    rects_offline_in_net = ax.bar(offline_pos, average_in_net_offline,
                                 # yerr=var_qoes_offline, ecolor='black', capsize=10,
                                  edgecolor=colors[1], color='white',
                                  # yerr=var_qoes_baloss, ecolor='black', capsize=10,
                                  hatch='/',
                                 bottom=y_offset,
                                 align='center', alpha=opaque, width=bar_width)
    y_offset = y_offset + average_in_net_offline
    rects_offline_in_netx = ax.bar(offline_pos, y_offset,
                                  # yerr=var_qoes_baloss, ecolor='black', capsize=10,
                                  edgecolor='black', color='none', linewidth=lwidth,
                                  # yerr=var_qoes_baloss, ecolor='black', capsize=10,
                                  align='center', width=bar_width)
    autolabel2(ax,offline_pos,y_offset,"OOS")
    autolabel2(ax, webrtc_pos+bar_width, np.zeros(len(y_offset)), "DASH")

    #autolabel(ax,loss_matrix_baloss, loss_matrix_webrtc)
    #xticks = [2,3,4,6,7]
    #xticks_minor = [1,5,6,7]
    baloss_ticks = LossAdaptConfigs.baloss_ticks
    #ax.set_xticks(xticks)
    #ax.set_xticks(xticks_minor, minor=True)
    #ax.set_xticklabels(baloss_ticks)
    #va = [0, -.05, 0, -.05, -.05, -.05]
    #for t, y in zip(ax.get_xticklabels(), va):
    #    t.set_y(y)
    #ax.set_xlim(1, 11)

    #ax.set_xlabel('Average QoE', fontsize=asize)
    ax.set_ylabel('Packet Loss(%)', fontsize=asize)
    #ax.set_ylabel('CDF', fontsize=asize)
    #ax.set_xlim(0, 0.85)
    ax.set_ylim(0, 8)
    plt.xticks(webrtc_pos+bar_width*1.5, baloss_ticks)
    plt.legend(loc='upper left', bbox_to_anchor=(0,1),ncol=1, handletextpad=0.1,
               borderaxespad=0.1,labelspacing=0.4,borderpad=0.1,
               columnspacing=0.5, fontsize=asize)
    plt.savefig(figname + ".pdf")
    #plt.savefig(figname + ".eps", type='eps')
    plt.show()

if __name__=="__main__":
    #the log ids of the with bandit
    log_nos_with = [1,2,3,5]
    #the log ids of those without bandit
    log_nos_without = [1]
    #the root path of the logs with bandit
    path_root_with = "log_with_bandit/"
    #the root path of the logs without bandit
    path_root_without = "log_without_bandit/"

    #the losss for first trace with bandit
    lossss_withs = []
    lossss_with_temp = []
    #iterate over all log_no for with bandit
    for log in log_nos_with:
        path_full_with = path_root_with+"r"+str(log)+".csv"
        #print("reading loss log:"+path_full_with)
        lossss_with_temp.append(ext_loss_from_qoe_log(path_full_with))
        lossss_withs.append((lossss_with_temp))
        lossss_with_temp = []

    # the losss for first trace without bandit
    lossss_withouts = []
    lossss_without_temp = []
    #iterate over all log_no for without bandit
    for log in log_nos_without:
        path_full_without = path_root_without + "r" + str(log) + "o.csv"
        #print("reading loss log:" + path_full_without)
        lossss_without_temp.append(ext_loss_from_qoe_log(path_full_without))
        lossss_withouts.append(lossss_without_temp)
        lossss_without_temp = []
    #print(lossss_withouts)
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['hatch.linewidth'] = 3
    # plt.rcParams['hatch.color'] = 'green'
    plt.rcParams["legend.handlelength"] = 1.0
    #plot the figure.1 comparing the log with different trace
    plot_bar_1([lossss_withs[0], lossss_withs[1], lossss_withs[2]],[lossss_withouts[0],lossss_withouts[0],lossss_withouts[0]],[lossss_withouts[0],lossss_withouts[0],lossss_withouts[0]],"loss_breakdown_bar")