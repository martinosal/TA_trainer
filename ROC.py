import h5py
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import copy



Nbins=1000

def integral(y,x):
    x_min=x
    x_max=1
    K=int(x_min*Nbins)
    s=0
    for i in range(K,Nbins):
        s=s+y[i]
    
    return s

class ROC:
    def __init__(self,MVA_model,X_test,y_test,MVA_model_name):
        noise_score_1=MVA_model.predict(X_test[np.where(y_test==1)])[:,0]
        noise_score_0=MVA_model.predict(X_test[np.where(y_test==0)])[:,0]
        signal_score_1=MVA_model.predict(X_test[np.where(y_test==1)])[:,1]
        signal_score_0=MVA_model.predict(X_test[np.where(y_test==0)])[:,1]

        y_signal,bins_1,_=plt.hist(signal_score_1, bins=Nbins, alpha=0.8, label='Signal')#, density=True
        y_noise,bins_1,_=plt.hist(signal_score_0, bins=Nbins, alpha=0.8, label='Background')#, density=True
        plt.yscale('log')
        plt.legend()
        plt.xlabel('MVA score')
        plt.savefig(MVA_model_name+'_score.pdf')
        plt.show()

        Nsignal=integral(y_signal,0)
        Nnoise=integral(y_noise,0)
        signal_eff=np.array([])
        noise_eff=np.array([])
        y_s=0
        y_n=0
        for i in range(0,Nbins+1):
            x=i/Nbins
            y_s=integral(y_signal,x)/Nsignal
            y_n=integral(y_noise,x)/Nnoise
            signal_eff=np.append(y_s,signal_eff)
            noise_eff=np.append(y_n,noise_eff)

        Area=round(1000*integral(signal_eff,0)/Nbins)/1000

        lab='Area: '+str(Area)
        plt.plot(noise_eff,signal_eff,label=lab)
        plt.plot([0,1],[0,1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        #plt.xlim([0.001,1])
        #plt.yscale('log')
        plt.title('ROC curve')
        plt.legend()
        plt.savefig('ROC_'+MVA_model_name+'.pdf')
        plt.show()

        WP=[0.90,0.94,0.97,0.99]
        rej=1./noise_eff
        WP_idx=[np.where(np.abs(signal_eff-WP[i])==np.min(np.abs(signal_eff-WP[i])))[0][0] for i in range(0,len(WP))]
        #rej[WP_idx]
        WP_rej=[str(round(10*rej[WP_idx[i]])/10) for i in range(0,len(WP))]
        print(WP_rej)

        plt.plot(signal_eff,rej)
        for i in range(0,len(WP)):
            plt.axvline(x=WP[i],color='Red',linestyle='dashed',label='Bkg Rejection @ '+str(WP[i])+' WP: '+WP_rej[i])
        plt.xlabel('Signal efficiency')
        plt.ylabel('Background rejection')
        plt.xlim([0.85,1])
        plt.yscale('log')
        plt.title('ROC curve')
        plt.legend()
        plt.savefig('rejection_ROC_'+MVA_model_name+'.pdf')
        plt.show()
        
        self.auc=Area
        self.signal_eff=signal_eff
        self.bkg_eff=noise_eff
        self.bkg_rej=rej
        self.WP=WP
        self.WP_rej=WP_rej