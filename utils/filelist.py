import os
filename=open("filename.txt","a+")

root_path='/lyxx/datasets/raw/ADD2023/test'
filelist=os.listdir(root_path)
filelist.sort()
for i in range(len(filelist)):
    filename.write(filelist[i])
    filename.write("\n")

