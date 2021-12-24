#created on 2021/01/12
#this runner call the scripts to draw all figures for ablation study

import os

def exec(script):
    cmd = "python3 " + script
    os.system(cmd)

if __name__=="__main__":
    prefix = "ablation_break_down_"
    affix = '.py'
    script_qoe = 'ablation_qoe.py'
    breakdown_types = ["delay","goodput","loss"]
    for breakdown_type in breakdown_types:
        script_name = prefix+breakdown_type+affix
        exec(script_name)
    exec(script_qoe)