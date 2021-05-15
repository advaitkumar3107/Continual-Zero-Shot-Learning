import os
import shutil
import time

def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


source_file = "/home/SharedData/fabio/data/UCF-101/split/train"
target_file = "/home/SharedData/fabio/zsl_cgan/ucf_split1"

train_test_split = 0.8

train_path = target_file + "/train"
test_path = target_file + "/test"

if os.path.exists(train_path):
    print("Deleting Existing Train Folder")
    shutil.rmtree(train_path)

time.sleep(2)

if os.path.exists(test_path):
    print("Deleting Existing Test Folder")
    shutil.rmtree(test_path)

time.sleep(2)
    
os.mkdir(train_path)
os.mkdir(test_path)

for (dir) in os.listdir(source_file):
    dir_name = source_file + '/' + dir
    train_path_name = train_path + '/' + dir
    test_path_name = test_path + '/' + dir
    os.mkdir(train_path_name)
    os.mkdir(test_path_name)
    train_length = train_test_split*len(os.listdir(dir_name))

    for i,dir1 in enumerate(os.listdir(dir_name)):
        dir_name1 = dir_name + '/' + dir1
        print(dir_name1)
        name1 = test_path + '/' + dir + '/' + dir1

        if (i < train_length): 
            name1 = train_path + '/' + dir + '/' + dir1

        os.mkdir(name1)            

        for (a, b, files) in os.walk(dir_name1):            
            for (i, file) in enumerate(files):
                file_name = dir_name1 + '/' + file
                shutil.copy(file_name, name1)