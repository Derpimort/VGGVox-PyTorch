#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:12:22 2020

@author: darp_lord
"""

import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler, SGD
from torch.utils.data import Subset, Dataset, DataLoader
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from sklearn.metrics import classification_report
import signal_utils as sig
from scipy.io import wavfile
from vggm import VGGM

DATA_DIR="../../../Projects/ML/VoxCeleb/"

LR=0.01
B_SIZE=1
N_EPOCHS=100
N_CLASSES=1251

class AudioDataset(Dataset):
    def __init__(self, csv_file):
        if isinstance(csv_file, str):
            csv_file=pd.read_csv(csv_file)
        assert isinstance(csv_file, pd.DataFrame), "Invalid csv path or dataframe"
        self.X=csv_file['Path'].values
        self.y=(csv_file['Label'].values-10000).astype(int)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        label=self.y[idx]
        sr, audio=wavfile.read(os.path.join(DATA_DIR+"wav",self.X[idx]))
        audio=sig.preprocess(audio)
        return torch.from_numpy(audio), torch.tensor(label)

    

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__=="__main__":
    df_meta=pd.read_csv(DATA_DIR+"vox1_meta.csv",sep="\t")
    df_F=pd.read_csv(DATA_DIR+"iden_split.txt", sep=" ", names=["Set","Path"] )
    df_F['Label']=df_F['Path'].str.split("/", n=1, expand=True)[0].str.replace("id","")
    print(df_F.head())
    df_F['Label']=df_F['Label'].astype(dtype=float)
    
    Datasets={
        "train":AudioDataset(df_F[df_F['Set']==1]),
        "val":AudioDataset(df_F[df_F['Set']==2]),
        "test":AudioDataset(df_F[df_F['Set']==3])}
    
    Dataloaders={i:DataLoader(Datasets[i], batch_size=B_SIZE, shuffle=False) for i in Datasets}
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    model=VGGM(N_CLASSES)
    loss_func=nn.CrossEntropyLoss()
    optimizer=SGD(model.parameters(), lr=LR, momentum=0.9)
    best_acc=1
    update_grad=128
    print("Start Training")
    for epoch in range(N_EPOCHS):
        running_loss=0.0
        loop=tqdm(Dataloaders['train'])
        
        for counter, (audio, labels) in enumerate(loop, start=1):
            audio=audio.unsqueeze(0)
            audio = audio.to(device)
            labels = labels.to(device)
            outputs = model(audio.float())
            loss = loss_func(outputs, labels)
            running_loss+=loss
            if(counter%update_grad==0):
                av_loss=running_loss/update_grad
                optimizer.zero_grad()
                av_loss.backward()
                optimizer.step()
                loop.set_postfix(loss=(running_loss))
                running_loss=0
                
        loop.set_description(f'Epoch [{epoch+1}/{N_EPOCHS}]')
            
        #print(f'Epoch [{epoch+1}/{N_EPOCHS}] Loss= {(running_loss/counter)}')
        model.eval()
        with torch.no_grad():
            y_pred=[]
            counter=0
            for audio, labels in Dataloaders['val']:
                audio = audio.to(device)
                labels = labels.to(device)
                outputs = model(audio)
                # max returns (value ,index)
                _, preds = torch.max(outputs, 1)
                y_pred+=preds.tolist()
                counter+=1
            acc1, acc5=accuracy(y_pred, Dataloaders['val'].targets, topk=(1,5))
            acc1/=counter
            acc5/=counter
            print("Val:\nTop-1 accuracy: %.5f, Top-5 accuracy: %.5f"%(acc1,acc5))
            if acc1<best_acc:
                best_acc=acc1
                torch.save(model.state_dict(), DATA_DIR+"/VGGM_%d.pth"%(epoch+1))
        model.train()
        scheduler.step()
        
    print('Finished Training..')
    PATH = DATA_DIR+"/VGGM_F.pth"
    torch.save(model.state_dict(), PATH)
