from sklearn.model_selection import train_test_split
import json
import random
import os
import pandas as pd


path = "../data/D2A_Dataset/D2A_func"
files = os.listdir(path)


vulnerableList = []
vul_counter = 0
ben_counter = 0


for file in files:


    with open("../data/D2A_Dataset/D2A_func/" + file) as f:
    #lines = f.readlines()
        
        for line in f:
            js=json.loads(line.strip())

            
            if js["label"] == 0:
                loc = ""
                continue
            else:
                #print("Type ...", type(str(js["bug_location"][0])))
                loc = str(js["bug_location"][0]) #+ "\""
                print(loc, "..............")
            vulnerableList.append({"processed_func": js["code"], "target": js["label"], "flaw_line_index": loc})
            
            if js["label"] == 0:
                ben_counter += 1
            else:
                vul_counter += 1


  
print("Length of vulnerable list ", len(vulnerableList))

finalList = vulnerableList


random.shuffle(finalList)
print("Len of benign code ", ben_counter)
print("Len of vulnerable list ", vul_counter)


"""

funcs = df["processed_func"].tolist()
        labels = df["target"].tolist()
        flaw_line_index = df["flaw_line_index"].tolist()
"""


#print(vulnerableList)

train, test = train_test_split(finalList, test_size=0.2)
val, test = train_test_split(test, test_size=0.5)



print("Total datasets ", vul_counter + ben_counter)



df_train = pd.DataFrame(train)
df_test = pd.DataFrame(test)
df_val = pd.DataFrame(val)

df_train.to_csv("train.csv")
df_test.to_csv("test.csv")
df_val.to_csv("val.csv")



