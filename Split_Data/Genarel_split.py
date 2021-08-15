import os
train=open("path_data/train_kvasir.txt","w+")
test=open("path_data/test_kvasir.txt","w+")
path_image="../Kvasir_SEG_Validation_120/images"
path_mask="../Kvasir_SEG_Validation_120/masks"
for count,i in enumerate(os.listdir(path_image)):
    image=path_image+'/'+i
    mask=path_mask+'/'+i
    string=image+"\t"+mask+"\n"
    if os.path.exists(mask):
        if count < 0.85* len(os.listdir(path_image)):
            train.write(string)
        else:
            test.write(string)