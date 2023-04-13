import os
import pandas as pd
import soundfile as sf
eval_list="/lyxx/datasets/raw/ADD2023/dev/label.txt"
eval_path="/lyxx/datasets/raw/ADD2023/dev/wav/"
new_path='/lyxx/datasets/raw/ADD2023/dev/newlabel.txt'



eval_list = pd.read_csv(eval_list, sep=' ', header=None)
for index in range(len(eval_list[0])):
    try:
        data_file_path = eval_path + eval_list.iloc[index, 0]
        audio, _ = sf.read(data_file_path)
        with open(new_path,"a+") as f:
            print(str(eval_list.iloc[index,0])+' '+str(eval_list.iloc[index,1]))
            f.write(str(eval_list.iloc[index,0])+' '+str(eval_list.iloc[index,1]))
            f.write("\n")
    except Exception as e:
        print(f"Error: {e}")
        continue
