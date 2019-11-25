import os

# train_dir = r'H:\AI4K\data\frame_data\training\HR'
val_dir = r'H:\AI4K\data\frame_data\validation\HR'
names = os.listdir(val_dir)

with open('data/validation.txt','w') as f:
    count = 1
    for name in names:
        print(str(count)+': '+name)
        f.writelines(name+'\n')
        count+=1

