import os
filename="result_new.txt"
othername="yql.txt"
filetest = open(filename, "r")
newfile=  open(othername,"a+")
file=filetest.read()
filelist=file.split("\n")
for score in range(len(filelist)):
    newscore=10.0-float(filelist[score])
    newfile.write(str(newscore))
    newfile.write("\n")