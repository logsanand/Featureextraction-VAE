import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plot_clusters(epochs,vae,data,path):

    fig, ax = plt.subplots()
    # ax.set_xticks(range(0, epochs+1, 50))
    ax.set(xlabel='z[0]', ylabel='z[1]',
           title=('Encoder clusters ='+ str(epochs)))
    ax.grid()
    z_mean = vae
    #ax.figure(figsize=(12, 10))
    ax.scatter(z_mean[:, 7], z_mean[:, 2])
    #ax.colorbar()
    fig.savefig(path+'clusters_'+str(epochs)+'.png')
    plt.close(fig)