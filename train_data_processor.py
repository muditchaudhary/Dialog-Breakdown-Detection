
# Python program to read
# json file

import os
import json

# Opening JSON file
for filename in os.listdir('dbdc4_en_dev_labeled/'):
#for filename in os.listdir('temp'):
    #content = f.read()
    #f = open(filename,)
    print("filename",filename)
    f = open('dbdc4_en_dev_labeled/'+filename,  'r+')
# returns JSON object as
# a dictionary
    data = json.load(f)

# Iterating through the json
# list
    k=0;
    for i in data['turns']:
    #print(i['annotations']);
        turns = []
        for j in i['annotations']:
            #print(j['breakdown']);
            turns.append(j['breakdown'])
        if turns :
            print(k);
            max2 = max(turns,key=turns.count)
            print("max ",max2)
            i['annotations']=max2
            with open('dbdc4_en_dev_labeled/'+filename, "w") as jsonFile:
                json.dump(data, jsonFile, indent=2)
        print("  ");
        k=k+1;

# Closing file
    f.close()
