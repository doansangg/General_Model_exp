import os
#train=open("path_data/train_kvasir.txt","w+")
test=open("path_data/test_kvasir_100.txt","w+")
path_image="../Kvasir_test/images"
path_mask="../Kvasir_test/masks"
for count,i in enumerate(os.listdir(path_image)):
    print(count)
    image="../Kvasir_test/images"+'/'+i
    mask="../Kvasir_test/masks"+'/'+i
    string=image+"\t"+mask+"\n"
    if os.path.exists(path_mask+'/'+i):
        # if count < 0.85* len(os.listdir(path_image)):
        #     train.write(string)
        # else:
        test.write(string)
