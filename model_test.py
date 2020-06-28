#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:12:22 2020

@author: darp_lord
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset, Dataset, DataLoader
from tqdm.auto import tqdm
from vggm import VGGM
import argparse
from train import AudioDataset, accuracy, ppdf, LOCAL_DATA_DIR, MODEL_DIR

def test(model, Dataloaders):
    corr1=0
    corr5=0
    counter=0
    top1=0
    top5=0
    for audio, labels in Dataloaders:
        audio = audio.to(device)
        labels = labels.to(device)
        outputs = model(audio)
        corr1, corr5=accuracy(outputs, labels, topk=(1,5))
        top1+=corr1
        top5+=corr5
    # max returns (value ,index)
        counter+=1
    print("\nTop-1 accuracy: %.5f\nTop-5 accuracy: %.5f"%(top1/counter, top5/counter))
    return top1/counter, top5/counter

if __name__=="__main__":
    parser=argparse.ArgumentParser(
        description="Train and evaluate VGGVox on complete voxceleb1 for identification")
    parser.add_argument("--dir","-d",help="Directory with wav and csv files", default="./Data/")
    args=parser.parse_args()
    DATA_DIR=args.dir 
    df_meta=pd.read_csv(LOCAL_DATA_DIR+"vox1_meta.csv",sep="\t")
    df_F=pd.read_csv(LOCAL_DATA_DIR+"iden_split.txt", sep=" ", names=["Set","Path"] )
    df_F=ppdf(df_F)
    
    Datasets={
        "val":AudioDataset(df_F[df_F['Set']==2], DATA_DIR, is_train=False),
        "test":AudioDataset(df_F[df_F['Set']==3], DATA_DIR, is_train=False)}
    Dataloaders={i:DataLoader(Datasets[i], batch_size=1, shuffle=False, num_workers=2) for i in Datasets}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    model=VGGM(1251)
    #model.load_state_dict(torch.load(DATA_DIR+"/VGGMVAL_BEST_149_80.84.pth", map_location=device))
    model.load_state_dict(torch.load(MODEL_DIR+"VGGM300_BEST_140_81.99.pth", map_location=device))
    model.to(device)
    model.eval()

    print("\nVal Score:\n")
    with torch.no_grad():
        acc1, _=test(model, Dataloaders['val'])

        
    print("\nTest Score:\n")
    acc1=test(model, Dataloaders['test'])

