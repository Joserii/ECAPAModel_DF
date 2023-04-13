import os


score = open("score.txt", "a+")
filename=open("filename.txt","r")
resultfile=open("result.txt","r")
file=filename.read()
filelist=file.split("\n")
result=resultfile.read()
resultlist=result.split("\n")
for i in range(len(filelist)):
    score.write(filelist[i])
    score.write("\n")
    score.write(resultlist[i])
    score.write("\n")
