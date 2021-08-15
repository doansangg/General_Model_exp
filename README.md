### 1.0 Requirements
  +  torch >= 1.4
  +  config  [gpu_devices = "0,1,2,3"] on file  ..._mutilgpu.py  
### 2.0. Download data TEST
  + Download data Train [download link (Google Drive)](https://mega.nz/file/LyZyxBzA#z69G9UYJI7eJm3DuJOyEpW2dOEzdus7tUfBnWrlF6Ss)
  + [test](https://mega.nz/file/2vYQxJCb#0JWEj8NKArWoo-63QlU6nCEA_5wFE2dea93YVytEGFY)
  + Downloading testing dataset[download link (Google Drive)](https://drive.google.com/file/d/17Cs2JhKOKwt4usiAYJVJMnXfyZWySn3s/view).
  + Downloading and move weight-backbone [harDNet68] (https://github.com/PingoLH/Pytorch-HarDNet)
### 3.0. Training

> Training model PraNet, Hardnet

1. Training model PraNet 
    > python3 MyTrain.py
    + mDICE = 0.904
    + mIOU = 0.848
1. Traing HardNet (original)
    > python3 Train_Main.py
    + mDICE = 0.912
    + mIOU = 0.857
1. Train HardNet_v2 (AttentionConv + Skip_Connection)
    > python3 Train_Main_v2.py
    + mDICE = 0.894
    + mIOU = 0.837
1. Train HardNet_v3 ( Create mask for 3 , cacualate Mask = 3 loss_funcition (0.1 , 0.3 , 0.6 ))
    > python3 Train_v3_mutilgpu.py 
    + mDICE = 0.903
    + mIOU = 0.846
1. Train HardNet_v4 (Skip_Connection)
    > python3 Train_v4_mutilgpu.py
    + mDICE = 0.896
    + mIOU = 0.839
1. Train HardNet_fpn (FPN)
    > python3 Train_fpn_mutilgpu.py
    + mDICE = 0.899
    + mIOU = 0.842

> Testing Configuration:
1. > python3 MyTest.py
> Eval 
1. > python3 MyEval.py 

### 4.0. CODE BK 
  >https://github.com/Rayicer/TransFuse
  + TransFuse-L 
  > mDCIE = 0.918       
  > mIOU = 0.868
  + TransFuse-S
  > mDCIE = 0.918       
  > mIOU = 	0.868
