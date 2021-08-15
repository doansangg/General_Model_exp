import os
train=open("path_data/train_kvasir.txt","w+")
test=open("path_data/test_kvasir_100.txt","r+")
list_test=test.readlines()
print(list_test)
path_image="../Kvasir-SEG/images"
path_mask="../Kvasir-SEG/masks"
for count,i in enumerate(os.listdir(path_image)):
    image=path_image+'/'+i
    mask=path_mask+'/'+i
    string=image+"\t"+mask+"\n"
    string_jpg=image.replace('.jpg','.png')+"\t"+mask.replace('.jpg','.png')+"\n"
    #print(string_jpg)
    if os.path.exists(mask):
        if string_jpg not in list_test:
            train.write(string)