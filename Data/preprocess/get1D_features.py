import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def get_all_seq_and_chain_name_and_seq_residues():
    all_H_and_L_chain = np.load("all_H_and_L_chain.npy",allow_pickle=True)
    all_H_and_L_chain = np.squeeze(all_H_and_L_chain)
    
    #获取每条链的名字
    all_chain_name = [ all_H_and_L_chain[i][0] 
                       for i in range(all_H_and_L_chain.shape[0]) ]
    
    #获取每条链上残基的标号（存在“30A 30B”这种情况）
    all_seq_residues = [ sorted(all_H_and_L_chain[i][1]) 
                         for i in range(all_H_and_L_chain.shape[0])]
    
    #获取所有序列（用于比对pssm长度是否与序列长度match）
    all_seq = [ [all_H_and_L_chain[i][1][res_id] for res_id in sorted(all_H_and_L_chain[i][1])] 
                 for i in range(all_H_and_L_chain.shape[0])]
        
    return all_seq, all_chain_name, all_seq_residues

#读取pssm特征
def read_pssm(fname,seq):
    num_pssm_cols = 44
    pssm_col_names = [str(j) for j in range(num_pssm_cols)]
    
    if os.path.exists(fname) == 0:
        print(fname)
        return np.zeros((120,20))
    
    with open(fname,'r') as f:
        tmp_pssm = pd.read_csv(f,delim_whitespace=True,names=pssm_col_names).dropna().values[:,2:22].astype(float)
        
    if tmp_pssm.shape[0] != len(seq):
        raise ValueError('PSSM file is in wrong format or incorrect!')
        
    return tmp_pssm

#读取hhm特征
def read_hhm(fname,seq):
    num_hhm_cols = 22
    hhm_col_names = [str(j) for j in range(num_hhm_cols)]
    
    if os.path.exists(fname) == 0:
        print(fname)
        return np.zeros((120,30))
    
    with open(fname,'r') as f:
        hhm = pd.read_csv(f,delim_whitespace=True,names=hhm_col_names)
    pos1 = (hhm['0']=='HMM').idxmax()+3
    num_cols = len(hhm.columns)
    hhm = hhm[pos1:-1].values[:,:num_hhm_cols].reshape([-1,44])
    hhm[hhm=='*']='9999'
    if hhm.shape[0] != len(seq):
        raise ValueError('HHM file is in wrong format or incorrect!')
        
    return hhm[:,2:-12].astype(float)

#spd33特征预处理
def spd3_feature_sincos(x,seq):
    ASA = x[:,0]
    rnam1_std = "ACDEFGHIKLMNPQRSTVWY-"
    ASA_std = (115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
                        185, 160, 145, 180, 225, 115, 140, 155, 255, 230,1)
    dict_rnam1_ASA = dict(zip(rnam1_std, ASA_std))
    ASA_div =  np.array([dict_rnam1_ASA[i] for i in seq])
    ASA = (ASA/ASA_div)[:,None]
    angles = x[:,1:5]
    HSEa = x[:,5:7]
    HCEprob = x[:,-3:]
    angles = np.deg2rad(angles)
    angles = np.concatenate([np.sin(angles),np.cos(angles)],1)
    return np.concatenate([ASA,angles,HSEa,HCEprob],1)

#读取spd33特征
def read_spd33(fname,seq):
    if os.path.exists(fname) == 0:
        print(fname)
        return np.zeros((120,14))
    
    with open(fname,'r') as f:
        spd3_features = pd.read_csv(f,delim_whitespace=True).values[:,3:].astype(float)
    tmp_spd3 = spd3_feature_sincos(spd3_features,seq)
    if tmp_spd3.shape[0] != len(seq):
        raise ValueError('SPD3 file is in wrong format or incorrect!')
    return tmp_spd3

#获取1D特征
def get1D_feature(all_seq, all_chain_name, all_seq_residues,tag):
    
    all_features = []

    for i in range(277):
        
        #抗体重链
        Heavy_chain_name = all_chain_name[i*2]
        feature_H1 = []
        feature_H2 = []
        feature_H3 = []
        if tag == "pssm":
            features = read_pssm("data/pssm/"+Heavy_chain_name+".pssm",all_seq[i*2])
        elif tag == "hhm":
            features = read_hhm("data/hhm/"+Heavy_chain_name+".hhm",all_seq[i*2])
        else:
            features = read_spd33("data/spd33/"+Heavy_chain_name+".spd33",all_seq[i*2])
        n = 0
        for index in range(len(all_seq_residues[i*2])):

            H_L_index = all_seq_residues[i*2][index][0]

            if( 24 <= H_L_index <= 34):
                feature_H1.append(features[index])

            elif(50<= H_L_index <= 58):
                feature_H2.append(features[index])

            elif(93<= H_L_index <=104):
                feature_H3.append(features[index])

            elif(H_L_index > 104):
                break

        feature_H1 = np.array(feature_H1)
        feature_H2 = np.array(feature_H2)
        feature_H3 = np.array(feature_H3)
        
        #抗体轻链
        Light_chain_name = all_chain_name[i*2+1]
        feature_L1 = []
        feature_L2 = []
        feature_L3 = []
        if tag == "pssm":
            features = read_pssm("data/pssm/"+Light_chain_name+".pssm",all_seq[i*2+1])
        elif tag == "hhm":
            features = read_hhm("data/hhm/"+Light_chain_name+".hhm",all_seq[i*2+1])
        else:
            features = read_spd33("data/spd33/"+Light_chain_name+".spd33",all_seq[i*2+1])
        n = 0
        for index in range(len(all_seq_residues[i*2+1])):

            H_L_index = all_seq_residues[i*2+1][index][0]

            if( 22 <= H_L_index <= 36):
                feature_L1.append(features[index])

            elif(48<= H_L_index <= 58):
                feature_L2.append(features[index])

            elif(87<= H_L_index <=99):
                feature_L3.append(features[index])

            elif(H_L_index > 99):
                break

        feature_L1 = np.array(feature_L1)
        feature_L2 = np.array(feature_L2)
        feature_L3 = np.array(feature_L3)


        all_features.append([Heavy_chain_name+str(1),feature_H1])
        all_features.append([Heavy_chain_name+str(2),feature_H2])
        all_features.append([Heavy_chain_name+str(3),feature_H3])
        all_features.append([Light_chain_name+str(1),feature_L1])
        all_features.append([Light_chain_name+str(2),feature_L2])
        all_features.append([Light_chain_name+str(3),feature_L3])
    
    return all_features

if __name__ == "__main__":
    all_seq, all_chain_name, all_seq_residues = get_all_seq_and_chain_name_and_seq_residues()
    pssm_feature = get1D_feature(all_seq, all_chain_name, all_seq_residues,"pssm")
    pssm_feature = np.save("1Dfeatures/pssm_init_features.npy",pssm_feature)
    hhm_feature = get1D_feature(all_seq, all_chain_name, all_seq_residues,"hhm")
    hhm_feature = np.save("1Dfeatures/hhm_init_features.npy",hhm_feature)
    spd33_feature = get1D_feature(all_seq, all_chain_name, all_seq_residues,"spd33")
    spd33_feature = np.save("1Dfeatures/spd33_init_features.npy",spd33_feature)
    
    
import copy

pssm_feature = np.load("1Dfeatures/pssm_init_features.npy",allow_pickle=True)
hhm_feature = np.load("1Dfeatures/hhm_init_features.npy",allow_pickle=True)
spd33_feature = np.load("1Dfeatures/spd33_init_features.npy",allow_pickle=True)

# print(len(pssm_feature))
# print(len(hhm_feature))
# print(len(spd33_feature))

all_pssm_feature = []
all_hhm_feature = []
all_spd33_feature = []

for i in range(len(pssm_feature)):
    pemp_pssm = copy.deepcopy(pssm_feature[i][1])
    all_pssm_feature.append(pemp_pssm)
        
    pemp_hhm = copy.deepcopy(hhm_feature[i][1])
    all_hhm_feature.append(pemp_hhm)
    
    pemp_spd33 = copy.deepcopy(spd33_feature[i][1])
    all_spd33_feature.append(pemp_spd33)


all_pssm_feature = np.array(all_pssm_feature)
all_hhm_feature = np.array(all_hhm_feature)
all_spd33_feature = np.array(all_spd33_feature)


#pssm特征归一化
guiyihua_pssm_feature = copy.deepcopy(all_pssm_feature)
all_pssm_feature_min = np.zeros(20)
all_pssm_feature_max = np.zeros(20)
all_pssm_feature_min[:] = 100000
all_pssm_feature_min

for i in range(all_pssm_feature.shape[0]):
    for j in range(20):
        all_pssm_feature_min[j] = min(all_pssm_feature[i][:,j].min() , all_pssm_feature_min[j])
        all_pssm_feature_max[j] = max(all_pssm_feature[i][:,j].max() , all_pssm_feature_max[j])

# print(all_pssm_feature_min)
# print(all_pssm_feature_max)

for i in range(all_pssm_feature.shape[0]):
    for j in range(20):
        guiyihua_pssm_feature[i][:,j] = all_pssm_feature[i][:,j] / (all_pssm_feature_max[j] - all_pssm_feature_min[j])

# guiyihua_pssm_feature.shape

#hhm特征归一化
guiyihua_hhm_feature = copy.deepcopy(all_hhm_feature)
all_hhm_feature_min = np.zeros(30)
all_hhm_feature_max = np.zeros(30)
all_hhm_feature_min[:] = 100000
all_hhm_feature_min

for i in range(all_hhm_feature.shape[0]):
    for j in range(30):
        all_hhm_feature_min[j] = min(all_hhm_feature[i][:,j].min() , all_hhm_feature_min[j])
        all_hhm_feature_max[j] = max(all_hhm_feature[i][:,j].max() , all_hhm_feature_max[j])

print(all_hhm_feature_min)
print(all_hhm_feature_max)

for i in range(all_hhm_feature.shape[0]):
    for j in range(30):
        guiyihua_hhm_feature[i][:,j] = all_hhm_feature[i][:,j] / (all_hhm_feature_max[j] - all_hhm_feature_min[j])

# guiyihua_hhm_feature.shape


#spd33特征归一化
guiyihua_spd33_feature = copy.deepcopy(all_spd33_feature)
all_spd33_feature_min = np.zeros(14)
all_spd33_feature_max = np.zeros(14)
all_spd33_feature_min[:] = 100000
all_spd33_feature_min

for i in range(all_spd33_feature.shape[0]):
    for j in range(14):
        all_spd33_feature_min[j] = min(all_spd33_feature[i][:,j].min() , all_spd33_feature_min[j])
        all_spd33_feature_max[j] = max(all_spd33_feature[i][:,j].max() , all_spd33_feature_max[j])

print(all_spd33_feature_min)
print(all_spd33_feature_max)

for i in range(all_spd33_feature.shape[0]):
    for j in range(14):
        guiyihua_spd33_feature[i][:,j] = all_spd33_feature[i][:,j] / (all_spd33_feature_max[j] - all_spd33_feature_min[j])

# guiyihua_spd33_feature.shape


max_length = 101
samples_num = 277

linked_pssm = np.zeros((samples_num,max_length,20))
linked_hhm = np.zeros((samples_num,max_length,30))
linked_spd33 = np.zeros((samples_num,max_length,14))

for i in range(samples_num):
    pemp_pssm = guiyihua_pssm_feature[i*6]
    pemp_hhm = guiyihua_hhm_feature[i*6]
    pemp_spd33 = guiyihua_spd33_feature[i*6]

    for j in range(1,6):
        pemp_pssm = np.append(pemp_pssm,np.zeros((1,20)),axis=0)
        pemp_pssm = np.append(pemp_pssm,guiyihua_pssm_feature[i*6+j],axis=0)
        pemp_hhm = np.append(pemp_hhm,np.zeros((1,30)),axis=0)
        pemp_hhm = np.append(pemp_hhm,guiyihua_hhm_feature[i*6+j],axis=0)
        pemp_spd33 = np.append(pemp_spd33,np.zeros((1,14)),axis=0)
        pemp_spd33 = np.append(pemp_spd33,guiyihua_spd33_feature[i*6+j],axis=0)

    
    if(pemp_pssm.shape[0] != pemp_hhm.shape[0]):
        print("---------------------------------")

    linked_pssm[i][0:pemp_pssm.shape[0],:] = pemp_pssm
    linked_hhm[i][0:pemp_hhm.shape[0],:] = pemp_hhm
    linked_spd33[i][0:pemp_spd33.shape[0],:] = pemp_spd33
    


dataset =  {
        "pssm": linked_pssm,
        "hhm": linked_hhm,
        "spd33": linked_spd33,
    }

np.save("train_1Dfeatures_linked.npy",dataset)
