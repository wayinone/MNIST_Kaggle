# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 18:16:50 2017

@author: wayin
"""
import numpy as np
import matplotlib.pyplot as plt
#%%

def plot_difficult_samples(model,x,y, verbose=True):
    """
    x: size(n,h,w,c)
    y: is categorical, i.e. onehot, size(n,p)
    """ 
    #%%
    
    pred_classes = model.predict_classes(x)
    y_val_classes = np.argmax(y, axis=1)
    er_id = np.nonzero(pred_classes!=y_val_classes)[0]
    #%%
    K = np.ceil(np.sqrt(len(er_id)))
    fig = plt.figure()
    print('There are %d wrongly predicted images out of %d validation samples'%(len(er_id),x.shape[0]))
    for i in range(len(er_id)):
        ax = fig.add_subplot(K,K,i+1)
        k = er_id[i]
        ax.imshow(x[er_id[i],:,:,0])
        ax.axis('off')
        if verbose:
            ax.set_title('%d as %d'%(y_val_classes[k],pred_classes[k]))
