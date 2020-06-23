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
from torch.optim import lr_scheduler, SGD, Adam
from torch.utils.data import Subset, Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import time
from tqdm.auto import tqdm
from sklearn.metrics import classification_report
import signal_utils as sig
from scipy.io import wavfile
from vggm import VGGM
import argparse



LR=0.01
B_SIZE=100
N_EPOCHS=30
N_CLASSES=1251
transformers=transforms.ToTensor()

class AudioDataset(Dataset):
    def __init__(self, csv_file, croplen=48320, is_train=True):
        if isinstance(csv_file, str):
            csv_file=pd.read_csv(csv_file)
        assert isinstance(csv_file, pd.DataFrame), "Invalid csv path or dataframe"
        self.X=csv_file['Path'].values
        self.y=(csv_file['Label'].values-10001).astype(int)
        self.is_train=is_train
        self.croplen=croplen
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        label=self.y[idx]
        sr, audio=wavfile.read(os.path.join(DATA_DIR,self.X[idx]))
        if(self.is_train):
            start=np.random.randint(0,audio.shape[0]-self.croplen+1)
            audio=audio[start:start+self.croplen]
        audio=sig.preprocess(audio).astype(np.float32)
        audio=np.expand_dims(audio, 2)
        return transformers(audio), label

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
            res.append((correct_k.mul(100.0 / batch_size)).item())
        return res
    
def test(model, Dataloader):
    corr1=0
    corr5=0
    counter=0
    top1=0
    top5=0
    for audio, labels in Dataloader:
        audio = audio.to(device)
        labels = labels.to(device)
        outputs = model(audio)
        corr1, corr5=accuracy(outputs, labels, topk=(1,5))
        top1+=corr1
        top5+=corr5
        # max returns (value ,index)
        counter+=1
    print("Val:\nTop-1 accuracy: %.5f\nTop-5 accuracy: %.5f"%(top1/counter, top5/counter))
    return top1/counter, top5/counter

if __name__=="__main__":
    parser=argparse.ArgumentParser(
        description="Train and evaluate VGGVox on complete voxceleb1 for identification")
    parser.add_argument("--dir","-d",help="Directory with wav and csv files", default="./Data/")
    args=parser.parse_args()
    DATA_DIR=args.dir 
    df_meta=pd.read_csv(DATA_DIR+"vox1_meta.csv",sep="\t")
    df_F=pd.read_csv(DATA_DIR+"iden_split.txt", sep=" ", names=["Set","Path"] )
    df_F['Label']=df_F['Path'].str.split("/", n=1, expand=True)[0].str.replace("id","")
    df_F['Label']=df_F['Label'].astype(dtype=float)
    # print(df_F.head(20))
    df_F['Path']="wav/"+df_F['Path']
    
    Datasets={
        "train":AudioDataset(df_F[df_F['Set']==1]),
        "val":AudioDataset(df_F[df_F['Set']==2], is_train=False),
        "test":AudioDataset(df_F[df_F['Set']==3], is_train=False)}
    batch_sizes={
            "train":B_SIZE,
            "val":1,
            "test":1}
    Dataloaders={i:DataLoader(Datasets[i], batch_size=batch_sizes[i], shuffle=True, num_workers=8) for i in Datasets}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    model=VGGM(1251)
    model.to(device)
    loss_func=nn.CrossEntropyLoss()
    optimizer=SGD(model.parameters(), lr=0.01)
    #scheduler=lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: LR[epoch])
    best_acc=1
    update_grad=1
    print("Start Training")
    for epoch in range(N_EPOCHS):
        model.train()
        #scheduler.step()
        running_loss=0.0
        corr1=0
        corr5=0
        top1=0
        top5=0
        random_subset=None
        loop=tqdm(Dataloaders['train'])
        loop.set_description(f'Epoch [{epoch+1}/{N_EPOCHS}]')
        for counter, (audio, labels) in enumerate(loop, start=1):
            optimizer.zero_grad()
            audio = audio.to(device)
            labels = labels.to(device)
            #print(labels.shape, audio.shape)
            #print(labels, audio[0])
            if counter==32:
                random_subset=audio
            outputs = model(audio)
            loss = loss_func(outputs, labels)
            running_loss+=loss
            corr1, corr5=accuracy(outputs, labels, topk=(1,5))
            top1+=corr1
            top5+=corr5
            if(counter%update_grad==0):
                #av_loss=running_loss/update_grad
                loss.backward()
                optimizer.step()
                loop.set_postfix(loss=(running_loss.item()/(counter)), top1_acc=top1/(counter), top5_acc=top5/counter)
                #running_loss=0

        #print(f'Epoch [{epoch+1}/{N_EPOCHS}] Loss= {(running_loss/counter)}')
        #scheduler.step()
        #print(audio.shape)
        model(random_subset)
        model.eval()
        with torch.no_grad():
            acc1, _=test(model, Dataloaders['val'])
            if acc1>best_acc:
                best_acc=acc1
                best_model=model.state_dict()

    torch.save(best_model, DATA_DIR+"/VGGM_BEST_%.2f.pth"%(acc1))
        
    print('Finished Training..')
    PATH = DATA_DIR+"/VGGM_F.pth"
    torch.save(model.state_dict(), PATH)
    model.eval()
    acc1=test(model, Dataloaders['test'])

