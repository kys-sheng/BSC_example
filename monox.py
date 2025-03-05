import pickle
import tarfile
import pandas as pd
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split
import time
import os
import sys
import random
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader,TensorDataset

# Check for available GPUs
if torch.cuda.is_available():
    # Choose the first available GPU
    device = torch.device("cuda")
    # Optionally, print information about each GPU
    num_gpus = torch.cuda.device_count()
    for gpu_id in range(num_gpus):
        print(f"GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
else:
    # Use the CPU if no GPU is available
    device = torch.device("cpu")
    print("No GPU available. Using CPU.")

print(device)

NBINS                 = 30
FILLED_BIN_PERCENTAGE = 0.5
XHIST                 = ['PT(j[1])', 'MET', 'ETA(j[1])'                           ]
XCUT                  = ['MET_CUT', 'ETA(j[1])_CUT'                               ]
XNEVE                 = ['PT(j[1])_NEVENTS', 'MET_NEVENTS', 'ETA(j[1])_NEVENTS'   ]
XRANGE                = ['PT(j[1])_LOW', 'MET_LOW', 'ETA(j[1])_UP' , 'PT(j[1])_UP', 'MET_UP' ]
XLAB                  = ['MASS', 'COUP'                                           ]
AUX_NAME              = ["MET_CUT" , "ETA(j[1])_CUT" , "PT(j[1])_NEVENTS" , "MET_NEVENTS" , "ETA(j[1])_NEVENTS" , "PT(j[1])_LOW" , "MET_LOW" , "ETA(j[1])_UP" , "PT(j[1])_UP" , "MET_UP"]
LAB_NAME              = ["MASS","COUP"]

def load_data(pattern,n=7):
    matching_files      = glob.glob(pattern)
    print(len(matching_files),len(matching_files)/n)
    data = pd.DataFrame()
    random.shuffle(matching_files)
    data_list = []
    for i in tqdm(matching_files):
        # load data.pkl
        with open(i, 'rb') as f:
            try:
                indat = pickle.load(f)       
                data_list.append(indat)

            except Exception as e:
                print(str(e))
                pass

    return pd.concat(data_list, axis=0)

def check_hist(x):
    assert all([isinstance(i,float) for i in x]), [type(i) for i in x]

def log_yscale(data,label=XHIST):
    idata = data.copy()
    for i in label:
        idata[i] = idata[i].progress_apply(lambda x : np.array([np.log10(i) if i != 0 else float(0) for i in x ]))        
        idata[i].progress_apply(check_hist)        
    return idata

def preprocess_noise_and_size(data,filled_bin_percentage=FILLED_BIN_PERCENTAGE):
    print("Enforced lower limit of bins coverage :",filled_bin_percentage)
    
    # Check for noise
    pre_len= len(data)
    print("Before Noise Filter:",pre_len)
    data["PT_noise"]  = data.progress_apply(lambda x: len([i for i in x["PT(j[1])"]  if i != 0])  / NBINS ,axis=1)
    #data["ETA_noise"] = data.progress_apply(lambda x: len([i for i in x["ETA(j[1])"] if i != 0])  / NBINS ,axis=1)
    data["MET_noise"] = data.progress_apply(lambda x: len([i for i in x["MET"]       if i != 0])  / NBINS ,axis=1)
    data              = data[(data["PT_noise"]    >= filled_bin_percentage) & (data["MET_noise"]   >= filled_bin_percentage)]
    print("After Noise Filter:",len(data))
    print("Filtered/Original %:",(len(data)/pre_len)*100)

    dat = []
    for i,idf in data.groupby(["MASS", "COUP"]):
        dat.append(len(idf))

    print(" Max Data Size per Parameter space point  :",np.array(dat).max())
    print(" Min Data Size per Parameter space point  :",np.array(dat).min())
    print(" Mean Data Size per Parameter space point :",np.array(dat).mean())
    print(" Std Data Size per Parameter space point  :",np.array(dat).std())

    new_dat = []
    new_df = []
    for i,idf in data.groupby(["MASS", "COUP"]):
        nsample = int(np.array(dat).min()*(random.uniform(1, 1.3)))
        try:
            sampled = idf.sample(nsample)
            
        except:
            sampled = idf.sample(np.array(dat).min())
        new_dat.append(len(sampled))
        new_df.append(sampled)

    data = pd.concat(new_df)
    print("After Size Standardization :",len(data))
    print("Cleaned/Original %         :",(len(data)/pre_len)*100)

    print(" New Max Data Size per Parameter space point  :",np.array(new_dat).max())
    print(" New Min Data Size per Parameter space point  :",np.array(new_dat).min())
    print(" New Mean Data Size per Parameter space point :",np.array(new_dat).mean())
    print(" New Std Data Size per Parameter space point  :",np.array(new_dat).std())
    
    return data



def save_pkl(stuff,name):
    with open(name, 'wb')  as file: 
        pickle.dump(stuff  , file)
    return      

# Define a custom function to check if the lists match
def lists_match(list1, list2):
    return list1 == list2


def preprocess(data):
    print("Data size: ",len(data))

    # Check for NaN values
    data = data.dropna()
    idata = data.copy()

    print("Data size without NaN: ",len(data))
   
    # Flatten range to be up and low columns
    for col in  data.columns:
        if "RANGE" in col:
            print("Rewrite: ",col)
            #data[i.replace("RANGE","LOW")] = data.progress_apply(lambda x: x[i][0],axis=1)
            #data[i.replace("RANGE","UP")] = data.progress_apply(lambda x:  x[i][1],axis=1)
            idata[col.replace("RANGE", "LOW")] = data[col].progress_apply(lambda x: x[0])
            idata[col.replace("RANGE", "UP")] = data[col].progress_apply(lambda x: x[1])
    
    # Taking only relevant columns
    data = idata[[k for k  in idata.columns if "RANGE" not in k]]
    #data = idata[['PT(j[1])','MET','ETA(j[1])','MET_CUT','ETA(j[1])_CUT','PT(j[1])_NEVENTS','MET_NEVENTS','ETA(j[1])_NEVENTS','PT(j[1])_LOW','MET_LOW','ETA(j[1])_LOW','PT(j[1])_UP','MET_UP','ETA(j[1])_UP','MASS','COUP']]

    # Number of duplicates
    print("Number of duplicates: ",data[data.describe().columns].duplicated().sum())
    
    # Remove duplicates    
    data[~data[data.describe().columns].duplicated()]
    return data

def plot_distribution(indata):
    # Exclude specific MASS and COUP values
    process_data_filtered = indata[(indata["MASS"] != 100) | (indata["COUP"] != 1.0)]

    # Get unique MASS-COUP combinations and their corresponding data sizes
    unique_combinations = process_data_filtered.groupby(["MASS", "COUP"]).size().reset_index(name="Data Size")

    # Print the unique combinations and their corresponding data sizes
    #print(unique_combinations)

    # Total Data Size and Unique MASS-COUP Combinations
    total_data_size = unique_combinations["Data Size"].sum()
    num_unique_combinations = len(unique_combinations)
    print("Total Data Size:", total_data_size)
    print("Unique MASS-COUP Combinations:", num_unique_combinations)

    # Extract MASS, COUP, and Data Size columns
    mass = unique_combinations["MASS"]
    coup = unique_combinations["COUP"]
    data_size = unique_combinations["Data Size"]

    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(mass, coup, c=data_size, cmap='viridis', alpha=0.9,s=100)
    plt.xlabel("Mass")
    plt.ylabel("Coupling")
    plt.title("Data Size per Mass and Coupling")
    plt.colorbar(label="Data Size")
    plt.grid(True)
    plt.show()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def make_hist_tensor(data,nchannel,nbins=NBINS):
    return torch.tensor(data.to_list()).reshape(len(data), nchannel, nbins)

 
class MyData():
    def __init__(self, data, nchannel,nbins,vsize=0.2,random_state=None,ref_dat=None,is_fsr=False):
        assert isinstance(data, pd.DataFrame)
        self.data                  = data.copy()
        self.ref_dat               = ref_dat
        self.vsize                 = vsize
        self.nchannel              = nchannel
        self.nbins                 = nbins
        self.is_fsr                = is_fsr

        input_data_len              = len(data)
        
        self.lab_name              = LAB_NAME         
        self.hist                  = self.data[XHIST]
        
        if not self.is_fsr : 
            self.aux_name              = AUX_NAME         
            self.other                 = self.data[XCUT + XNEVE + XRANGE + XLAB]
        else:
            self.other                 = self.data[XLAB]
        

        self.get_global_norm()
        self.normalize(ref_dat)
        normed_data_len             = len(self.data)
        
        if not self.is_fsr :  
            self.aux_name_normed       = [i+"_NORM" for i in self.aux_name]
        self.lab_name_normed       = [i+"_NORM" for i in self.lab_name]

        self.other_normed.columns  = [i+"_NORM" for i in self.other.columns]
        self.hist_g.columns        = [i+"_G" for i in self.hist_g.columns]
        self.hist_l.columns        = [i+"_L" for i in self.hist_l.columns]

        self.other                 = pd.concat([self.other_normed,self.other],axis=1)
        self.hist                  = pd.concat([self.hist_l,self.hist_g,self.hist],axis=1)

        self.else_pd =  self.data[[i for i in self.data.columns if (i not in self.other.columns) and (i not in self.hist.columns)]]
        print("self.else_pd.columns     :",self.else_pd.columns)


        if len(self.else_pd.columns)>=1:
            self.data                  = pd.concat([self.hist,self.other,self.else_pd],axis=1)
        else:
            self.data                  = pd.concat([self.hist,self.other],axis=1)
            
        concat_data_len            = len(self.data)

        self.data = self.repackage_hist_data(self.data)
        repackage_data_len          = len(self.data)

        print("Train Data Size :",input_data_len, normed_data_len, concat_data_len,repackage_data_len)
        assert input_data_len == normed_data_len == concat_data_len == repackage_data_len, "Data size mismatch"
        
        self.train_data, self.valid_data = train_test_split(self.data,test_size=vsize,shuffle=True,random_state=random_state)

    
    def get_global_norm(self):
        self.global_norm_dict    = {}
        for i in self.hist.columns:
            self.global_norm_dict[i] = {}
            self.global_norm_dict[i]["max"] = self.hist.progress_apply(lambda x: np.max(x[i]),axis=1).max()
            self.global_norm_dict[i]["min"] = self.hist.progress_apply(lambda x: np.min(x[i]),axis=1).min()
        return self.global_norm_dict    

    def lnorm(self,x):
        if x.max()-x.min() != 0   : k = (x-x.min())/(x.max()-x.min())
        else                      : k = x
        return k

    def normalize(self,ref_dat=None):
        # For fsr data
        if ref_dat is None: ref_dat = self

        # normalize and check for NaN values
        self.other_normed   = (self.other - ref_dat.other[self.other.columns].min()   )    / (ref_dat.other[self.other.columns].max()  - ref_dat.other[self.other.columns].min() )
        if self.other_normed.isnull().values.any():   
            raise ValueError("NaN values in the other_normed")
        
        # Normalize (GLOBAL)
        self.hist_g = self.hist.copy() 
        self.hist_g = self.hist_g.progress_apply(lambda x: (x-ref_dat.global_norm_dict[x.name]["min"])/(ref_dat.global_norm_dict[x.name]["max"]-ref_dat.global_norm_dict[x.name]["min"]),axis=0)
        
        # Normalize (LOCAL)
        self.hist_l = self.hist.copy()
        for i in self.hist.columns: self.hist_l[i] = self.hist.progress_apply(lambda x: self.lnorm(x[i]),axis=1)
    
    def repackage_hist_data(self,indata):
        indata["INPUT_G"] = indata.progress_apply(lambda x: (np.array([x["PT(j[1])_G"].astype(float),x["MET_G"].astype(float),x["ETA(j[1])_G"].astype(float)])), axis=1)
        indata["INPUT_L"] = indata.progress_apply(lambda x: (np.array([x["PT(j[1])_L"].astype(float),x["MET_L"].astype(float),x["ETA(j[1])_L"].astype(float)])), axis=1)
        return indata

    def get_fsr(self, met_cut, eta_cut, pt_range, met_range, eta_up, norm_parent=True):
        
        self.fsr_data = self.data.copy()
        
        print("Getting Fixed Signal Region(FSR)...")
        self.fsr_data = self.fsr_data[ (self.fsr_data["MET_CUT"] == met_cut) ]
        
        print("After cut (MET_CUT)   \t:",len(self.fsr_data))
        if len(self.fsr_data) == 0: raise ValueError("No self.fsr_data found after cuts MET_CUT")

        self.fsr_data = self.fsr_data[ (self.fsr_data["ETA(j[1])_CUT"] == eta_cut) ]
        
        print("After cut (ETA_CUT)   \t:",len(self.fsr_data))
        if len(self.fsr_data) == 0: raise ValueError("No self.fsr_data found after cuts ETA_CUT")

        self.fsr_data = self.fsr_data[self.fsr_data["PT(j[1])_LOW"].progress_apply(lambda x: x == pt_range[0])]
        self.fsr_data = self.fsr_data[self.fsr_data["PT(j[1])_UP"].progress_apply(lambda x: x == pt_range[1])]
        
        print("After cut (PT_RANGE)  \t:",len(self.fsr_data))
        if len(self.fsr_data) == 0: raise ValueError("No self.fsr_data found after cuts PT_RANGE")

        self.fsr_data = self.fsr_data[self.fsr_data["MET_LOW"].progress_apply(lambda x: x == met_range[0])]
        self.fsr_data = self.fsr_data[self.fsr_data["MET_UP"].progress_apply(lambda x: x == met_range[1])]
        
        print("After cut (MET_RANGE) \t:",len(self.fsr_data))
        if len(self.fsr_data) == 0: raise ValueError("No self.fsr_data found after cuts MET_RANGE")

        
        self.fsr_data = self.fsr_data[self.fsr_data["ETA(j[1])_UP"].progress_apply(lambda x: lists_match(x, eta_up))]
        
        print("After cut (ETA_RANGE) \t:",len(self.fsr_data))
        if len(self.fsr_data) == 0: raise ValueError("No self.fsr_data found after cuts ETA_RANGE")
        if norm_parent  : self.fsr = MyData(self.fsr_data,self.nchannel,self.nbins,is_fsr=False,ref_dat=self)
        else            : self.fsr = MyData(self.fsr_data,self.nchannel,self.nbins,is_fsr=False,ref_dat=None)    


def make_train_data(input_path,output_path):
    print("Making Training Data Obj...")
    print("input_path   :",input_path)
    print("output_path  :",output_path)
    input_data     =  load_pkl(input_path)
    data_classobj  = MyData(input_data,3,30)
    data_classobj.get_fsr(met_cut=120, eta_cut=4.5, pt_range=(120,1500.000), met_range=(120,1500.000), eta_up=4.5)
    print("Saving Training Data Obj...")
    save_pkl(data_classobj , output_path)
    return 




def make_other_data(input_path,ref_path,get_fsr,output_path):

    print("Making Other Data Obj...")
    print("input_path   :",input_path)
    print("ref_path     :",ref_path)
    print("output_path  :",output_path)
    
    print("get_fsr      :",get_fsr)
    
    ref_classobj   = load_pkl(ref_path)
    input_data     = load_pkl(input_path)

    out_classobj  = MyData(input_data,3,30,ref_dat=ref_classobj)
    if get_fsr: out_classobj.get_fsr(met_cut=120, eta_cut=4.5, pt_range=(120,1500.000), met_range=(120,1500.000), eta_up=4.5)
    print("Saving Other Data Obj...")
    save_pkl(out_classobj      , output_path)  
    return out_classobj    

def prepare_dataset_for_training(outpath,dataset,test_data=None,test_data_fsr=None,batch_size=512,norm="g",nchannel=3,nbin=30,scale="lin"):
    assert isinstance(dataset,MyData)
    assert isinstance(test_data,type(pd.DataFrame()))      or test_data is None
    assert isinstance(test_data_fsr,type(pd.DataFrame())) or test_data_fsr is None
    
    if test_data is not None : 
        test_data = MyData(test_data.copy(),ref_dat=dataset,nchannel=dataset.nchannel,nbins=dataset.nbins)
        if test_data_fsr is not None : 
            test_data_fsr = MyData(test_data_fsr.copy(),ref_dat=dataset,nchannel=dataset.nchannel,nbins=dataset.nbins)
            test_data_fsr = test_data_fsr.data
        test_data = test_data.data

    train_data    = dataset.train_data
    val_data      = dataset.valid_data
    val_data_fsr  = dataset.fsr.data

    if norm == "g":
        train_hist                           = make_hist_tensor(train_data["INPUT_G"],nchannel,nbin).to(device)
        val_hist                             = make_hist_tensor(val_data["INPUT_G"],nchannel,nbin).to(device)
        val_hist_fsr                         = make_hist_tensor(val_data_fsr["INPUT_G"],nchannel,nbin).to(device)
    elif norm == "l":
        train_hist                           = make_hist_tensor(train_data["INPUT_L"],nchannel,nbin).to(device)
        val_hist                             = make_hist_tensor(val_data["INPUT_L"],nchannel,nbin).to(device)
        val_hist_fsr                         = make_hist_tensor(val_data_fsr["INPUT_L"],nchannel,nbin).to(device)
    else: raise KeyError

    train_aux                            = torch.tensor(train_data[dataset.aux_name].values).to(device)
    val_aux                              = torch.tensor(val_data[dataset.aux_name].values).to(device)
    val_aux_fsr                          = torch.tensor(val_data_fsr[dataset.aux_name].values).to(device)
        
    train_aux_normed                     = torch.tensor(train_data[dataset.aux_name_normed].values).to(device)
    val_aux_normed                       = torch.tensor(val_data[dataset.aux_name_normed].values).to(device)
    val_aux_fsr_normed                   = torch.tensor(val_data_fsr[dataset.aux_name_normed].values).to(device)
        
        
    train_lab                            = torch.tensor(train_data[dataset.lab_name].values).to(device)
    val_lab                              = torch.tensor(val_data[dataset.lab_name].values).to(device)
    val_lab_fsr                           = torch.tensor(val_data_fsr[dataset.lab_name].values).to(device)

    train_lab_normed                     = torch.tensor(train_data[dataset.lab_name_normed].values).to(device)
    val_lab_normed                       = torch.tensor(val_data[dataset.lab_name_normed].values).to(device)
    val_lab_fsr_normed                   = torch.tensor(val_data_fsr[dataset.lab_name_normed].values).to(device)
        
    print("Hist Shape          :", train_hist.shape          , val_hist.shape       , val_hist_fsr.shape       )
    print("Aux Shape           :", train_aux.shape           , val_aux.shape        , val_aux_fsr.shape        )
    print("Aux Shape(normed)   :", train_aux_normed.shape    , val_aux_normed.shape , val_aux_fsr_normed.shape )
    print("Lab Shape           :", train_lab.shape           , val_lab.shape        , val_lab_fsr.shape        )
    print("Lab Shape(normed)   :", train_lab_normed.shape    , val_lab_normed.shape , val_lab_fsr_normed.shape )

    test_hist                            = None 
    test_hist_fsr                        = None 
    if test_data is not None : 
        if norm == "g": test_hist                        = make_hist_tensor(test_data["INPUT_G"],nchannel,nbin).to(device)
        if norm == "l": test_hist                        = make_hist_tensor(test_data["INPUT_L"],nchannel,nbin).to(device)
        test_aux                         = torch.tensor(test_data[dataset.aux_name].values).to(device)            
        test_aux_normed                  = torch.tensor(test_data[dataset.aux_name_normed].values).to(device)            
        test_lab                         = torch.tensor(test_data[dataset.lab_name].values).to(device)    
        test_lab_normed                  = torch.tensor(test_data[dataset.lab_name_normed].values).to(device)      
        print("TEST hist Shape       :",test_hist.shape)
        print("TEST aux Shape        :",test_aux.shape)
        print("TEST aux_normed Shape :",test_aux_normed.shape)
        print("TEST lab Shape        :",test_lab.shape)
        print("TEST lab_normed Shape :",test_lab_normed.shape)
        
    if test_data_fsr is not None : 
        if norm == "g": test_hist_fsr                        = make_hist_tensor(test_data_fsr["INPUT_G"],nchannel,nbin).to(device)
        if norm == "l": test_hist_fsr                        = make_hist_tensor(test_data_fsr["INPUT_L"],nchannel,nbin).to(device)
        test_aux_fsr                         = torch.tensor(test_data_fsr[dataset.aux_name].values).to(device)            
        test_aux_normed_fsr                  = torch.tensor(test_data_fsr[dataset.aux_name_normed].values).to(device)            
        test_lab_fsr                         = torch.tensor(test_data_fsr[dataset.lab_name].values).to(device)    
        test_lab_normed_fsr                  = torch.tensor(test_data_fsr[dataset.lab_name_normed].values).to(device)      
        print("TEST(fsr) hist Shape       :",test_hist_fsr.shape)
        print("TEST(fsr) aux Shape        :",test_aux_fsr.shape)
        print("TEST(fsr) aux_normed Shape :",test_aux_normed_fsr.shape)
        print("TEST(fsr) lab Shape        :",test_lab_fsr.shape)
        print("TEST(fsr) lab_normed Shape :",test_lab_normed_fsr.shape)

    tensor_dataset     = TensorDataset(train_hist.to(device)   , train_aux.to(device)   , train_aux_normed.to(device)   , train_lab.to(device)   , train_lab_normed.to(device))
    dataloader         = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)

    save_pkl(dataloader            , f"{outpath}/dataloader_{scale}_{norm}_{batch_size}.pkl")                        
    save_pkl(train_hist            , f"{outpath}/train_hist_{scale}_{norm}_{batch_size}.pkl")                        
    save_pkl(train_aux             , f"{outpath}/train_aux_{scale}_{norm}_{batch_size}.pkl")                        
    save_pkl(train_aux_normed      , f"{outpath}/train_aux_normed_{scale}_{norm}_{batch_size}.pkl")                        
    save_pkl(train_lab             , f"{outpath}/train_lab_{scale}_{norm}_{batch_size}.pkl")                        
    save_pkl(train_lab_normed      , f"{outpath}/train_lab_normed_{scale}_{norm}_{batch_size}.pkl")                        
    save_pkl(val_hist              , f"{outpath}/val_hist_{scale}_{norm}_{batch_size}.pkl")                        
    save_pkl(val_aux               , f"{outpath}/val_aux_{scale}_{norm}_{batch_size}.pkl")                        
    save_pkl(val_aux_normed        , f"{outpath}/val_aux_normed_{scale}_{norm}_{batch_size}.pkl")                        
    save_pkl(val_lab               , f"{outpath}/val_lab_{scale}_{norm}_{batch_size}.pkl")                        
    save_pkl(val_lab_normed        , f"{outpath}/val_lab_normed_{scale}_{norm}_{batch_size}.pkl")                        
    save_pkl(val_hist_fsr          , f"{outpath}/val_hist_fsr_{scale}_{norm}_{batch_size}.pkl")                        
    save_pkl(val_aux_fsr           , f"{outpath}/val_aux_fsr_{scale}_{norm}_{batch_size}.pkl")                        
    save_pkl(val_aux_fsr_normed    , f"{outpath}/val_aux_fsr_normed_{scale}_{norm}_{batch_size}.pkl")                        
    save_pkl(val_lab_fsr           , f"{outpath}/val_lab_fsr_{scale}_{norm}_{batch_size}.pkl")                        
    save_pkl(val_lab_fsr_normed    , f"{outpath}/val_lab_fsr_normed_{scale}_{norm}_{batch_size}.pkl")                        
    save_pkl(test_hist             , f"{outpath}/test_hist_{scale}_{norm}_{batch_size}.pkl")                        
    save_pkl(test_aux              , f"{outpath}/test_aux_{scale}_{norm}_{batch_size}.pkl")                        
    save_pkl(test_aux_normed       , f"{outpath}/test_aux_normed_{scale}_{norm}_{batch_size}.pkl")                        
    save_pkl(test_lab              , f"{outpath}/test_lab_{scale}_{norm}_{batch_size}.pkl")                        
    save_pkl(test_lab_normed       , f"{outpath}/test_lab_normed_{scale}_{norm}_{batch_size}.pkl")                        
    save_pkl(test_hist_fsr         , f"{outpath}/test_hist_fsr_{scale}_{norm}_{batch_size}.pkl")                        
    save_pkl(test_aux_fsr          , f"{outpath}/test_aux_fsr_{scale}_{norm}_{batch_size}.pkl")                        
    save_pkl(test_aux_normed_fsr   , f"{outpath}/test_aux_normed_fsr_{scale}_{norm}_{batch_size}.pkl")                        
    save_pkl(test_lab_fsr          , f"{outpath}/test_lab_fsr_{scale}_{norm}_{batch_size}.pkl")                        
    save_pkl(test_lab_normed_fsr   , f"{outpath}/test_lab_normed_fsr_{scale}_{norm}_{batch_size}.pkl")       

    return (dataloader,
           train_hist   , train_aux   , train_aux_normed   , train_lab   , train_lab_normed   , 
           val_hist     , val_aux     , val_aux_normed     , val_lab     , val_lab_normed     , 
           val_hist_fsr , val_aux_fsr , val_aux_fsr_normed , val_lab_fsr , val_lab_fsr_normed , 
           test_hist    , test_aux    , test_aux_normed    , test_lab    , test_lab_normed    , 
           test_hist_fsr, test_aux_fsr, test_aux_normed_fsr, test_lab_fsr, test_lab_normed_fsr     )

def make_dataset(traindata_classobj_path,testdata_classobj_path,testdata_classobj_fsr_path,output_path,scale,batch_size=512):
    traindata_classobj    = load_pkl(traindata_classobj_path)                               
    testdata              = load_pkl(testdata_classobj_path)                               
    testdata              = testdata.data
    testdata_fsr          = load_pkl(testdata_classobj_fsr_path)    
    testdata_fsr          = testdata_fsr.data

    prepare_dataset_for_training(output_path,
                                        traindata_classobj,
                                        test_data=testdata,
                                        test_data_fsr=testdata_fsr,
                                        batch_size=batch_size,
                                        norm="g",
                                        scale=scale
                                        )
    prepare_dataset_for_training(output_path,
                                        traindata_classobj,
                                        test_data=testdata,
                                        test_data_fsr=testdata_fsr,
                                        batch_size=batch_size,
                                        norm="l",
                                        scale=scale)

def load_dataset(inpath,scale,norm,batch_size):
    dataloader           = load_pkl( f"{inpath}/dataloader_{scale}_{norm}_{batch_size}.pkl")                        
    train_hist           = load_pkl( f"{inpath}/train_hist_{scale}_{norm}_{batch_size}.pkl")                        
    train_aux            = load_pkl( f"{inpath}/train_aux_{scale}_{norm}_{batch_size}.pkl")                        
    train_aux_normed     = load_pkl( f"{inpath}/train_aux_normed_{scale}_{norm}_{batch_size}.pkl")                        
    train_lab            = load_pkl( f"{inpath}/train_lab_{scale}_{norm}_{batch_size}.pkl")                        
    train_lab_normed     = load_pkl( f"{inpath}/train_lab_normed_{scale}_{norm}_{batch_size}.pkl")                        
    val_hist             = load_pkl( f"{inpath}/val_hist_{scale}_{norm}_{batch_size}.pkl")                        
    val_aux              = load_pkl( f"{inpath}/val_aux_{scale}_{norm}_{batch_size}.pkl")                        
    val_aux_normed       = load_pkl( f"{inpath}/val_aux_normed_{scale}_{norm}_{batch_size}.pkl")                        
    val_lab              = load_pkl( f"{inpath}/val_lab_{scale}_{norm}_{batch_size}.pkl")                        
    val_lab_normed       = load_pkl( f"{inpath}/val_lab_normed_{scale}_{norm}_{batch_size}.pkl")                        
    val_hist_fsr         = load_pkl( f"{inpath}/val_hist_fsr_{scale}_{norm}_{batch_size}.pkl")                        
    val_aux_fsr          = load_pkl( f"{inpath}/val_aux_fsr_{scale}_{norm}_{batch_size}.pkl")                        
    val_aux_fsr_normed   = load_pkl( f"{inpath}/val_aux_fsr_normed_{scale}_{norm}_{batch_size}.pkl")                        
    val_lab_fsr          = load_pkl( f"{inpath}/val_lab_fsr_{scale}_{norm}_{batch_size}.pkl")                        
    val_lab_fsr_normed   = load_pkl( f"{inpath}/val_lab_fsr_normed_{scale}_{norm}_{batch_size}.pkl")                        
    test_hist            = load_pkl( f"{inpath}/test_hist_{scale}_{norm}_{batch_size}.pkl")                        
    test_aux             = load_pkl( f"{inpath}/test_aux_{scale}_{norm}_{batch_size}.pkl")                        
    test_aux_normed      = load_pkl( f"{inpath}/test_aux_normed_{scale}_{norm}_{batch_size}.pkl")                        
    test_lab             = load_pkl( f"{inpath}/test_lab_{scale}_{norm}_{batch_size}.pkl")                        
    test_lab_normed      = load_pkl( f"{inpath}/test_lab_normed_{scale}_{norm}_{batch_size}.pkl")                        
    test_hist_fsr        = load_pkl( f"{inpath}/test_hist_fsr_{scale}_{norm}_{batch_size}.pkl")                        
    test_aux_fsr         = load_pkl( f"{inpath}/test_aux_fsr_{scale}_{norm}_{batch_size}.pkl")                        
    test_aux_normed_fsr  = load_pkl( f"{inpath}/test_aux_normed_fsr_{scale}_{norm}_{batch_size}.pkl")                        
    test_lab_fsr         = load_pkl( f"{inpath}/test_lab_fsr_{scale}_{norm}_{batch_size}.pkl")                        
    test_lab_normed_fsr  = load_pkl( f"{inpath}/test_lab_normed_fsr_{scale}_{norm}_{batch_size}.pkl")       

    return (dataloader,
           train_hist   , train_aux   , train_aux_normed   , train_lab   , train_lab_normed   , 
           val_hist     , val_aux     , val_aux_normed     , val_lab     , val_lab_normed     , 
           val_hist_fsr , val_aux_fsr , val_aux_fsr_normed , val_lab_fsr , val_lab_fsr_normed , 
           test_hist    , test_aux    , test_aux_normed    , test_lab    , test_lab_normed    , 
           test_hist_fsr, test_aux_fsr, test_aux_normed_fsr, test_lab_fsr, test_lab_normed_fsr     )



def load_model(modelobj,name,device):
    model = (modelobj).to(device)
    model.load_state_dict(torch.load(name,map_location=device))  
    return model


# Function to print GPU memory usage
def gpu_usage():
    allocated_memory = torch.cuda.memory_allocated()
    total_memory = torch.cuda.get_device_properties(0).total_memory
    cached_memory = torch.cuda.memory_reserved()

    allocated_percentage = (allocated_memory / total_memory) * 100
    cached_percentage = (cached_memory / total_memory) * 100

    return allocated_percentage,cached_percentage

# Function to compute the total norm of gradients
def compute_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def train_aux(model,loader,valid_dat_tnsr,valid_aux_tnsr,valid_lab_tnsr,valid_dat_tnsr_mr,valid_aux_tnsr_mr,valid_lab_tnsr_mr,best_model_path,nepochs = 1000 ,patience=50,learning_rate=0.001,max_norm=500,plot=False):
    start             = time.time()
    optimizer         = optim.Adam(model.parameters(), lr=learning_rate)
    history_valid     = []
    history           = []
    history_valid_mr  = []
    history_max_grad  = []
    history_grad      = []
    best_val_loss     = float('inf')

    model.train()
    model.to("cpu")
    model.pdf.init_params(data=valid_lab_tnsr.to("cpu"))
    model.to(device)
    
    with tqdm(range(nepochs)) as epoch_pbar:
        for epoch in epoch_pbar:
        # Train the model for one epoch
            model.train()
            rerun          = 0
            while True:
                average_loss              = 0
                average_grad              = 0
                max_grad                  = 0
                smooth_run                = True

                for inputs, _, aux_normed, _, targets_normed in loader:
                    try:
                        optimizer.zero_grad()
                        outputs                = model(inputs.to(device),aux_normed.to(device)).to(device)
                        log_pdf,_,_            = model.pdf(targets_normed.to(device), conditional_input=outputs)
                        neg_log_loss           = -log_pdf.mean() 
                        neg_log_loss.backward()
                        
                        # Calculate gradient norm before clipping
                        grad_norm_before = compute_grad_norm(model)
                        if max_grad < grad_norm_before: max_grad = grad_norm_before
        
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        
                        # Calculate gradient norm after clipping
                        grad_norm_after = compute_grad_norm(model)
                        
                        if grad_norm_before != grad_norm_after: print(grad_norm_before,grad_norm_after)
                        optimizer.step()

                    except Exception as e:
                        print("Failed batch run: ",str(e))
                        smooth_run = False
                        break

                    average_loss += neg_log_loss.item()
                    average_grad += grad_norm_before
                average_loss = average_loss / len(loader)    
                average_grad = average_grad / len(loader)    
                
                if smooth_run == True:  break
                
                rerun += 1
                print("Rerunning:",rerun)
                if rerun == 1                                                     : load_name = best_model_path
                elif os.path.exists(best_model_path.replace(".pth", "_prev.pth")) : load_name = best_model_path.replace(".pth", "_prev.pth")
                else                                                              : load_name = best_model_path

                try                   : 
                    model.load_state_dict(torch.load(load_name,map_location=device))  
                except Exception as e : 
                    print(f"Failed Model Load {load_name}: ",str(e))
                    print(grad_norm_before,grad_norm_after)
                if rerun > 3          : raise KeyError("Reran 3 times still cant work")
            
            # Evaluate the model on the valid set
            model.eval()
            with torch.no_grad():
                valid_outputs = model(valid_dat_tnsr.to(device),valid_aux_tnsr.to(device))
                log_pdf,_,_   = model.pdf(valid_lab_tnsr.to(device), conditional_input=valid_outputs)
                valid_loss    = -log_pdf.mean()
                
                valid_outputs_mr = model(valid_dat_tnsr_mr.to(device),valid_aux_tnsr_mr.to(device))
                log_pdf_mr,_,_   = model.pdf(valid_lab_tnsr_mr.to(device), conditional_input=valid_outputs_mr)
                valid_loss_mr    = -log_pdf_mr.mean()
                if plot:
                  target_sample, base_sample, target_log_pdf, base_log_pdf  = model.pdf.sample(conditional_input=valid_outputs.to(device)[0].repeat(50000,1))                        
                  xx,yy = np.transpose(torch.tensor(target_sample).cpu())
                  xx,yy = np.array(xx),np.array(yy)
                  counts, x_edges, y_edges, _ = plt.hist2d(xx,yy,bins=(75,75))
                  peak_index = np.unravel_index(np.argmax(counts), counts.shape)
                  peak_x = (x_edges[peak_index[0]] + x_edges[peak_index[0] + 1]) / 2
                  peak_y = (y_edges[peak_index[1]] + y_edges[peak_index[1] + 1]) / 2
                  plt.title(f"TRUTH: {str(valid_lab_tnsr.cpu().clone().detach().tolist()[0])} Pred: {peak_x},{peak_y}")
                  plt.xlim(0,1)
                  plt.ylim(0,1)
                  plt.show()
                  plt.figure(figsize=(3,3))
                  plt.title("xx")
                  plt.hist(xx,bins=90)
                  plt.xlim(0,1)
                  plt.show()
                  plt.figure(figsize=(3,3))            
                  plt.title("yy")
                  plt.hist(yy,bins=90)
                  plt.xlim(0,1)
                  plt.show()


           # Save the loss history
            history.append(average_loss)
            history_valid.append(float(valid_loss))
            history_valid_mr.append(float(valid_loss_mr))
            history_max_grad.append(max_grad)
            history_grad.append(average_grad)
            if plot:
                plt.figure(figsize=(3,3))
                plt.plot(history)
                plt.plot(history_valid)
                plt.plot(history_valid_mr)
                plt.show()
                plt.figure(figsize=(3,3))
                plt.plot(history_max_grad)
                plt.plot(history_grad)
                plt.show()
            
            if valid_loss < best_val_loss:
                best_val_loss    = valid_loss
                current_patience = 0
                if os.path.exists(best_model_path): os.rename(best_model_path, best_model_path.replace(".pth", "_prev.pth"))
                torch.save(model.state_dict(), best_model_path)
                best_epoch       = epoch            

            else:
                current_patience += 1

            # Early stopping check
            if current_patience >= patience:
                print(f'Validation loss has not improved for {patience} epochs. Early stopping at {epoch}...')
                break    
 
            epoch_pbar.set_postfix({"Train Loss": history[-5:], "Val Loss": history_valid[-5:], "Val Loss (MR)": history_valid_mr[-5:],"max_grad":history_max_grad[-5:],"ave_grad":history_grad[-5:]})
 
        end        = time.time()
        best_model = deepcopy(model)
        best_model.load_state_dict(torch.load(best_model_path))
        
        print("took {} seconds ".format(end-start))
        print("best_val_loss   :",best_val_loss)        
        print("best_epoch      :",best_epoch)        

        return model,best_model,history,history_valid,history_valid_mr,history_max_grad, history_grad, best_epoch



def train_no_aux(model,loader,valid_dat_tnsr,valid_lab_tnsr,valid_dat_tnsr_mr,valid_lab_tnsr_mr,best_model_path,nepochs = 1000 ,patience=50,learning_rate=0.001,max_norm=500,plot=False):
    start             = time.time()
    optimizer         = optim.Adam(model.parameters(), lr=learning_rate)
    history_valid     = []
    history           = []
    history_valid_mr  = []
    history_max_grad  = []
    history_grad      = []
    best_val_loss     = float('inf')

    model.train()
    model.to("cpu")
    model.pdf.init_params(data=valid_lab_tnsr.to("cpu"))
    model.to(device)
    
    with tqdm(range(nepochs)) as epoch_pbar:
        for epoch in epoch_pbar:
        # Train the model for one epoch
            model.train()
            rerun          = 0
            while True:
                average_loss              = 0
                average_grad              = 0
                max_grad                  = 0
                smooth_run                = True

                for inputs, _, _ , _, targets_normed in loader:
                    try:
                        optimizer.zero_grad()
                        outputs                = model(inputs.to(device)).to(device)
                        log_pdf,_,_            = model.pdf(targets_normed.to(device), conditional_input=outputs)
                        neg_log_loss           = -log_pdf.mean() 
                        neg_log_loss.backward()
                        
                        # Calculate gradient norm before clipping
                        grad_norm_before = compute_grad_norm(model)
                        if max_grad < grad_norm_before: max_grad = grad_norm_before
        
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        
                        # Calculate gradient norm after clipping
                        grad_norm_after = compute_grad_norm(model)
                        
                        if grad_norm_before != grad_norm_after: print(grad_norm_before,grad_norm_after)
                        optimizer.step()

                    except Exception as e:
                        print("Failed batch run: ",str(e))
                        smooth_run = False
                        break

                    average_loss += neg_log_loss.item()
                    average_grad += grad_norm_before
                average_loss = average_loss / len(loader)    
                average_grad = average_grad / len(loader)    
                
                if smooth_run == True:  break
                
                rerun += 1
                print("Rerunning:",rerun)
                if rerun == 1                                                     : load_name = best_model_path
                elif os.path.exists(best_model_path.replace(".pth", "_prev.pth")) : load_name = best_model_path.replace(".pth", "_prev.pth")
                else                                                              : load_name = best_model_path

                try                   : 
                    model.load_state_dict(torch.load(load_name,map_location=device))  
                except Exception as e : 
                    print(f"Failed Model Load {load_name}: ",str(e))
                    print(grad_norm_before,grad_norm_after)
                if rerun > 3          : raise KeyError("Reran 3 times still cant work")
            
            # Evaluate the model on the valid set
            model.eval()
            with torch.no_grad():
                valid_outputs = model(valid_dat_tnsr.to(device))
                log_pdf,_,_   = model.pdf(valid_lab_tnsr.to(device), conditional_input=valid_outputs)
                valid_loss    = -log_pdf.mean()
                
                valid_outputs_mr = model(valid_dat_tnsr_mr.to(device))
                log_pdf_mr,_,_   = model.pdf(valid_lab_tnsr_mr.to(device), conditional_input=valid_outputs_mr)
                valid_loss_mr    = -log_pdf_mr.mean()
                if plot:
                  target_sample, base_sample, target_log_pdf, base_log_pdf  = model.pdf.sample(conditional_input=valid_outputs.to(device)[0].repeat(50000,1))                        
                  xx,yy = np.transpose(torch.tensor(target_sample).cpu())
                  xx,yy = np.array(xx),np.array(yy)
                  counts, x_edges, y_edges, _ = plt.hist2d(xx,yy,bins=(75,75))
                  peak_index = np.unravel_index(np.argmax(counts), counts.shape)
                  peak_x = (x_edges[peak_index[0]] + x_edges[peak_index[0] + 1]) / 2
                  peak_y = (y_edges[peak_index[1]] + y_edges[peak_index[1] + 1]) / 2
                  plt.title(f"TRUTH: {str(valid_lab_tnsr.cpu().clone().detach().tolist()[0])} Pred: {peak_x},{peak_y}")
                  plt.xlim(0,1)
                  plt.ylim(0,1)
                  plt.show()
                  plt.figure(figsize=(3,3))
                  plt.title("xx")
                  plt.hist(xx,bins=90)
                  plt.xlim(0,1)
                  plt.show()
                  plt.figure(figsize=(3,3))            
                  plt.title("yy")
                  plt.hist(yy,bins=90)
                  plt.xlim(0,1)
                  plt.show()


           # Save the loss history
            history.append(average_loss)
            history_valid.append(float(valid_loss))
            history_valid_mr.append(float(valid_loss_mr))
            history_max_grad.append(max_grad)
            history_grad.append(average_grad)
            if plot:
                plt.figure(figsize=(3,3))
                plt.plot(history)
                plt.plot(history_valid)
                plt.plot(history_valid_mr)
                plt.show()
                plt.figure(figsize=(3,3))
                plt.plot(history_max_grad)
                plt.plot(history_grad)
                plt.show()
            
            if valid_loss < best_val_loss:
                best_val_loss    = valid_loss
                current_patience = 0
                if os.path.exists(best_model_path): os.rename(best_model_path, best_model_path.replace(".pth", "_prev.pth"))
                torch.save(model.state_dict(), best_model_path)
                best_epoch       = epoch            

            else:
                current_patience += 1

            # Early stopping check
            if current_patience >= patience:
                print(f'Validation loss has not improved for {patience} epochs. Early stopping at {epoch}...')
                break    
 
            epoch_pbar.set_postfix({"Train Loss": history[-5:], "Val Loss": history_valid[-5:], "Val Loss (MR)": history_valid_mr[-5:],"max_grad":history_max_grad[-5:],"ave_grad":history_grad[-5:]})
 
        end        = time.time()
        best_model = deepcopy(model)
        best_model.load_state_dict(torch.load(best_model_path))
        
        print("took {} seconds ".format(end-start))
        print("best_val_loss   :",best_val_loss)        
        print("best_epoch      :",best_epoch)        

        return model,best_model,history,history_valid,history_valid_mr,history_max_grad, history_grad, best_epoch


def run_train(inpath,scale,norm,batch_size,jammy_layers,nepochs,patience,model_type="rawcnn"):
    # load_dataset(inpath,scale,norm,batch_size)
    ( x_dataloader,
      x_train_hist   , x_train_aux   , x_train_aux_normed   , x_train_lab   , x_train_lab_normed   , 
      x_val_hist     , x_val_aux     , x_val_aux_normed     , x_val_lab     , x_val_lab_normed     , 
      x_val_hist_fsr , x_val_aux_fsr , x_val_aux_fsr_normed , x_val_lab_fsr , x_val_lab_fsr_normed , 
      x_test_hist    , x_test_aux    , x_test_aux_normed    , x_test_lab    , x_test_lab_normed    , 
      x_test_hist_fsr, x_test_aux_fsr, x_test_aux_normed_fsr, x_test_lab_fsr, x_test_lab_normed_fsr     ) = mx.load_dataset(inpath,scale,norm,batch_size)

    flow_options_overwrite=dict()
    if "g" in jammy_layers:
        flow_options_overwrite["g"]=dict()
        flow_options_overwrite["g"]["upper_bound_for_widths"]=1
        flow_options_overwrite["g"]["lower_bound_for_widths"]=0.01
        flow_options_overwrite["g"]["fit_normalization"]=0
        
    flow_options_overwrite["t"]=dict()
    flow_options_overwrite["t"]["cov_type"]="full"

    tag = f"test_run_{jammy_layers}_{scale}_{norm}_{nepochs}_{patience}_{model_type}"
    if model_type =="rawcnn":
      (rawcnn_monox_model,
       rawcnn_monox_best_model,
       rawcnn_monox_history,
       rawcnn_monox_history_val,
       rawcnn_monox_history_val_mr,
       rawcnn_monox_history_max_grad,
       rawcnn_monox_history_grad,
       rawcnn_monox_best_epoch) = train_aux((RAWCNN(aux_input_size=10,cond_dim=10, hidden_size=30,nchannel=3,nlast=18,jammy_layers=jammy_layers,flow_options_overwrite=flow_options_overwrite).double()).to(device)       , 
                                                                     x_dataloader               , 
                                                                     x_val_hist                 , 
                                                                     x_val_aux_normed           , 
                                                                     x_val_lab_normed           , 
                                                                     x_val_hist_fsr             , 
                                                                     x_val_aux_fsr_normed       , 
                                                                     x_val_lab_fsr_normed       , 
                                                                     f"{dir_path}/proper_run/rawcnn_monox_best_model_{tag}.pth",
                                                                     nepochs  =nepochs,
                                                                     patience =patience,
                                                                     plot=True
                                                                   )
      torch.save(rawcnn_monox_model.state_dict() , f"{dir_path}/proper_run/rawcnn_monox_model_{tag}.pth" )
      mx.save_pkl(rawcnn_monox_history           , f"{dir_path}/proper_run/rawcnn_monox_history_{tag}.pkl" )
      mx.save_pkl(rawcnn_monox_history_val       , f"{dir_path}/proper_run/rawcnn_monox_history_val_{tag}.pkl" )
      mx.save_pkl(rawcnn_monox_history_val_mr    , f"{dir_path}/proper_run/rawcnn_monox_history_val_mr_{tag}.pkl" )
      mx.save_pkl(rawcnn_monox_history_max_grad  , f"{dir_path}/proper_run/rawcnn_monox_history_max_grad_{tag}.pkl" )    
      mx.save_pkl(rawcnn_monox_history_grad      , f"{dir_path}/proper_run/rawcnn_monox_history_grad_{tag}.pkl" )    
      mx.save_pkl(rawcnn_monox_best_epoch        , f"{dir_path}/proper_run/rawcnn_monox_best_epoch_{tag}.pkl" )      
      print("max_grad        : ", max(rawcnn_monox_history_max_grad))
      print("best_epoch      : ", rawcnn_monox_best_epoch)
      print() 

    elif model_type =="cnn":
      (cnn_monox_model,
       cnn_monox_best_model,
       cnn_monox_history,
       cnn_monox_history_val,
       cnn_monox_history_val_mr,
       cnn_monox_history_max_grad,
       cnn_monox_history_grad,
       cnn_monox_best_epoch) = train_no_aux((CNN(cond_dim=10, nchannel=3,nlast=18,jammy_layers=jammy_layers,flow_options_overwrite=flow_options_overwrite).double()).to(device)       , 
                                                                     x_dataloader               , 
                                                                     x_val_hist                 , 
                                                                     x_val_lab_normed           , 
                                                                     x_val_hist_fsr             , 
                                                                     x_val_lab_fsr_normed       , 
                                                                     f"{dir_path}/proper_run/cnn_monox_best_model_{tag}.pth",
                                                                     nepochs  =nepochs,
                                                                     patience =patience
                                                                   )
      torch.save(cnn_monox_model.state_dict() , f"{dir_path}/proper_run/cnn_monox_model_{tag}.pth" )
      mx.save_pkl(cnn_monox_history           , f"{dir_path}/proper_run/cnn_monox_history_{tag}.pkl" )
      mx.save_pkl(cnn_monox_history_val       , f"{dir_path}/proper_run/cnn_monox_history_val_{tag}.pkl" )
      mx.save_pkl(cnn_monox_history_val_mr    , f"{dir_path}/proper_run/cnn_monox_history_val_mr_{tag}.pkl" )
      mx.save_pkl(cnn_monox_history_max_grad  , f"{dir_path}/proper_run/cnn_monox_history_max_grad_{tag}.pkl" )    
      mx.save_pkl(cnn_monox_history_grad      , f"{dir_path}/proper_run/cnn_monox_history_grad_{tag}.pkl" )    
      mx.save_pkl(cnn_monox_best_epoch        , f"{dir_path}/proper_run/cnn_monox_best_epoch_{tag}.pkl" )      
      print("max_grad        : ", max(cnn_monox_history_max_grad))
      print("best_epoch      : ", cnn_monox_best_epoch)
      print() 
    else:
      raise KeyError


def load_pkl(name):
    with open(name, 'rb') as file: 
        return pickle.load(file)       


def load_pkl_from_targz(tar_gz_path, pickle_filename):
    with tarfile.open(tar_gz_path, "r:gz") as tar:
        with tar.extractfile(pickle_filename) as file:
            return pickle.load(file)