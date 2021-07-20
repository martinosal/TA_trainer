import h5py
import math
import numpy as np
from numpy import genfromtxt
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib.pyplot as plt
from matplotlib import gridspec
import copy

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential

from sklearn.decomposition import PCA, KernelPCA

def Sig_to_Bkg_eq(signal,bkg,p):
    y=np.concatenate((np.zeros(len(bkg)),np.ones(len(signal))))
    X=np.concatenate((bkg,signal))

    X, y = shuffle(X, y, random_state=0)

    X_sig=X[np.where(y==1)]
    X_bkg=X[np.where(y==0)]

#    p=0.4
    Min=min(X_sig.shape[0],X_bkg.shape[0])
    N=int(Min/p)
    #print(N)
    if Min==X_sig.shape[0]:
        X_bkg=X_bkg[:N]
    if Min==X_bkg.shape[0]:
        X_sig=X_sig[:N]

    print('new bkg size: ',X_bkg.shape,'; new signal size: ',X_sig.shape,'; new S/B ratio: ',X_sig.shape[0]/X_bkg.shape[0])
    X_cut=np.concatenate((X_bkg,X_sig))
    y_cut=np.concatenate((np.zeros(len(X_bkg)),np.ones(len(X_sig))))

    X, y = shuffle(X_cut, y_cut, random_state=0)
    return X,y
    
    
#Index(['trk_log_pt', 'trk_eta', 'phi', 'theta', 'log_dr', 'log_ptfrac']
#cut_str='pTfrac<10 and pT>1e3'
def preprocess_DF(data, cut_str):
    data_cp=data.copy()
    if cut_str!='':
        data_cp=data_cp.query(cut_str)
        
    data_cp['pT']=data_cp['pT'].apply(lambda x: math.log(x))
    data_cp=data_cp.rename(columns={'pT': 'trk_log_pt'})
    data_cp=data_cp.rename(columns={'Eta': 'trk_eta'})
    data_cp=data_cp.rename(columns={'Phi': 'phi'})
    data_cp=data_cp.rename(columns={'Theta': 'theta'})
    data_cp['DR']=data_cp['DR'].apply(lambda x: math.log(x))
    data_cp=data_cp.rename(columns={'DR': 'log_dr'})
    data_cp['pTfrac']=data_cp['pTfrac'].apply(lambda x: math.log(x))
    data_cp=data_cp.rename(columns={'pTfrac': 'log_ptfrac'})
    return data_cp
   
def sig_bkg_selector(data,drop_list):
    signal=data.query('isfromBD==1')#tracks from BD
    bkg=data.query('isfromBD==0')#other tracks

#    drop_list=['isfromBD']
    signal=signal.drop(drop_list, 1)
    bkg=bkg.drop(drop_list, 1)
    #columns_list=signal.columns

    print('bkg size: ',bkg.shape, '; signal size: ',signal.shape, '; S/B ratio: ',signal.shape[0]/bkg.shape[0])
    return signal,bkg
    
def plot_sig_bkg(X,y,columns_list,density_f,mode,path):
    fig=plt.figure(figsize=(16,8))
    gs = gridspec.GridSpec(nrows=2, ncols=3, width_ratios=[1,1,1],figure=fig)
#    fig.suptitle('No cut')

    for i in range(0,len(columns_list)):
        ax=fig.add_subplot(gs[i])
        im = ax.hist(X[np.where(y==0),i][0],bins=100,alpha=0.5,density=density_f,label='bkg')
        im = ax.hist(X[np.where(y==1),i][0],bins=100,alpha=0.5,density=density_f,label='signal')
        ax.set_xlabel(columns_list[i])
        if mode=='log':
            ax.set_yscale('log')
        ax.legend()

    plt.savefig(path+'vars_sig_bkg'+mode+'.pdf')
    plt.show()
    
    
def plot_train_test(X_train,X_test,columns_list,density_f,mode,path):
    fig=plt.figure(figsize=(16,8))
    gs = gridspec.GridSpec(nrows=2, ncols=3, width_ratios=[1,1,1],figure=fig)

    for i in range(0,len(columns_list)):
        ax=fig.add_subplot(gs[i])
        im = ax.hist(X_train[:,i],bins=100,alpha=0.5,density=density_f,label='train')
        im = ax.hist(X_test[:,i],bins=100,alpha=0.5,density=density_f,label='test')
        ax.set_xlabel(columns_list[i])
        if mode=='log':
            ax.set_yscale('log')
        ax.legend()

    plt.savefig(path+'vars_train_test'+mode+'.pdf')
    plt.show()

#path ../TA_RootTree_to_DR/datasets/
def dataset_split(X,y,test_frac,path):
#    test_frac=0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac)

    rscaler_train = RobustScaler().fit(X_train)
    X_train_scaled=rscaler_train.transform(X_train)

#    rscaler_test = RobustScaler().fit(X_test)
    X_test_scaled=rscaler_train.transform(X_test)

    y_train_cat=to_categorical(y_train)
    y_test_cat=to_categorical(y_test)
    
    np.savetxt(path+"X_train.csv",X_train,delimiter=',')
    np.savetxt(path+"X_train_scaled.csv",X_train_scaled,delimiter=',')
    np.savetxt(path+"X_test.csv",X_test,delimiter=',')
    np.savetxt(path+"X_test_scaled.csv",X_test_scaled,delimiter=',')
    np.savetxt(path+"y_train.csv",y_train,delimiter=',')
    np.savetxt(path+"y_train_cat.csv",y_train_cat,delimiter=',')
    np.savetxt(path+"y_test.csv",y_test,delimiter=',')
    np.savetxt(path+"y_test_cat.csv",y_test_cat,delimiter=',')

    print('saved: X_train, X_train_scaled, X_test, X_test_scaled, y_train, y_train_cat, y_test, y_test_cat')
    print('in: ',path)
    
    return X_train, X_train_scaled, X_test, X_test_scaled, y_train, y_train_cat, y_test, y_test_cat

def retrieve_dataset(path):
    X_train=genfromtxt(path+"X_train.csv",delimiter=',')
    X_train_scaled=genfromtxt(path+"X_train_scaled.csv",delimiter=',')
    X_test=genfromtxt(path+"X_test.csv",delimiter=',')
    X_test_scaled=genfromtxt(path+"X_test_scaled.csv",delimiter=',')
    y_train=genfromtxt(path+"y_train.csv",delimiter=',')
    y_train_cat=genfromtxt(path+"y_train_cat.csv",delimiter=',')
    y_test=genfromtxt(path+"y_test.csv",delimiter=',')
    y_test_cat=genfromtxt(path+"y_test_cat.csv",delimiter=',')
    return X_train, X_train_scaled, X_test, X_test_scaled, y_train, y_train_cat, y_test, y_test_cat


def masked_data(X,y,mask):
    
    mask=[0,1,4,5]
    X_repeated_0=np.array([])
    y_repeated_0=np.array([])
    X_repeated_0=X[:,mask][np.where(y==0)]
    y_repeated_0=y[np.where(y==0)]

    X_repeated_1=np.array([])
    y_repeated_1=np.array([])
    X_repeated_1=X[:,mask][np.where(y==1)]
    y_repeated_1=y[np.where(y==1)]

    X_repeated=np.vstack((X_repeated_0,X_repeated_1))
    y_repeated=np.append(y_repeated_0,y_repeated_1)

    X_repeated,y_repeated=shuffle(X_repeated,y_repeated, random_state=0)
    y_repeated_cat=to_categorical(y_repeated)

    
    print(X_repeated_0.shape,X_repeated_1.shape)
    print(X_repeated.shape,y_repeated.shape,y_repeated_cat.shape)
    return X_repeated,y_repeated,y_repeated_cat


def bHjet_DR_jetPt(data_bH,path):
    jet_pt=1e-3*data_bH['bH_pT']/data_bH['bH_ptfrac']
    fig, ax = plt.subplots(1,figsize=(8,4))
    im = ax.hist2d(jet_pt,data_bH['bH_DR'],bins=(100,100))
    plt.xlim([0.,2000])
    plt.xlabel('jet pT [GeV]')
    plt.ylabel('DR(jet,bH)')
    fig.colorbar(im[3], ax=ax,aspect=8)
    plt.savefig(path+'bHjet_DR_jetPt.pdf')
    plt.show()
    
    
def sig_bkg_DR_jetPt(trk_df,path):
    jet_pt_sig=trk_df.query('isfromBD==1')['pT']/trk_df.query('isfromBD==1')['pTfrac']
    jet_pt_bkg=trk_df.query('isfromBD==0')['pT']/trk_df.query('isfromBD==0')['pTfrac']

    fig, ax = plt.subplots(1,figsize=(8,4))
    im = ax.hist2d(jet_pt_sig,trk_df.query('isfromBD==1')['DR'],bins=(100,100))
    plt.xlim([0.,2000])
    plt.xlabel('jet pT [GeV]')
    plt.ylabel('DR(jet,trk)')
    fig.colorbar(im[3], ax=ax,aspect=8)
    plt.savefig(path+'sig_DR_jetPt.pdf')
    plt.show()
    
    fig, ax = plt.subplots(1,figsize=(8,4))
    im = ax.hist2d(jet_pt_bkg,trk_df.query('isfromBD==0')['DR'],bins=(100,100))
    plt.xlim([0.,2000])
    plt.xlabel('jet pT [GeV]')
    plt.ylabel('DR(jet,trk)')
    fig.colorbar(im[3], ax=ax,aspect=8)
    plt.savefig(path+'bkg_DR_jetPt.pdf')
    plt.show()
    
    
def bH_plot(bH_df,path):
    fig=plt.figure(figsize=(16,8))
    gs = gridspec.GridSpec(nrows=2, ncols=3, width_ratios=[1,1,1],figure=fig)
#    fig.suptitle('No cut')
    columns_list=bH_df.columns
    for i in range(0,len(columns_list)):
        ax=fig.add_subplot(gs[i])
        im = ax.hist(bH_df[columns_list[i]],bins=100,alpha=0.5)#,density=density_f)
        ax.set_xlabel(columns_list[i])
        ax.set_yscale('log')
        #ax.legend()
    plt.savefig(path+'bH_plots.pdf')