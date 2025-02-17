import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch


l = [4,5]
t = [4,8,12]
for i in l:
    for j in t:

            fpath = f'explain_res/cmnist/GIN_layer_{i}/res_dict_top{j}.pt'
            res = torch.load(fpath,map_location=torch.device('cpu'))
            print (f'layer {i} top {j}')
            for k in res:
                print (res[k][1])
            
            

