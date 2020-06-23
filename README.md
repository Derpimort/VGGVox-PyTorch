# VGGVox-PyTorch
Implementing VGGVox for VoxCeleb1 dataset in PyTorch.

###### Specify data dir with --dir
```python3 train.py --dir ./Data/```

## Notes
- Couldn't replicate the results, only got 70% top-1, 88% top-5 at best on the variable input network. 
- Still off by 10% atleast. 
- Training on the V100 takes 4 mins per epoch.

#### What i've done so far:
 - [x] All the data preprocessed exactly as author's matlab code. Checked and verified online on matlab
 - [x] Random 3s cropped segments for training.
 - [x] Copy all hyperparameter... LR, optimizer params, batch size from the author's net.
 - [x] Stabilize PyTorch's BatchNorm and test variants. Improved results by a small percentage.
 - [x] Try onesided spectrogram input as mentioned on the author's github.
 - [ ] Port the authors network from matlab and train. The matlab model has 1300 outputs dimension, will test it later.
 - [ ] Copy weights from the matlab network and test.

# References and Citations:

 - [VGGVox](https://github.com/a-nagrani/VGGVox)
 - linhdvu14's [vggvox-speaker-identification](https://github.com/linhdvu14/vggvox-speaker-identification)
 - jameslyons's [python_speech_features](https://github.com/jameslyons/python_speech_features)
 
 ```bibtex
@InProceedings{Nagrani17,
  author       = "Nagrani, A. and Chung, J.~S. and Zisserman, A.",
  title        = "VoxCeleb: a large-scale speaker identification dataset",
  booktitle    = "INTERSPEECH",
  year         = "2017",
}


@InProceedings{Nagrani17,
  author       = "Chung, J.~S. and Nagrani, A. and Zisserman, A.",
  title        = "VoxCeleb2: Deep Speaker Recognition",
  booktitle    = "INTERSPEECH",
  year         = "2018",
}
```

