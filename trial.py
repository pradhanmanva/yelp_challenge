from sklearn.model_selection import RepeatedKFold
import pandas as pd
import os
import csv

business_df = pd.read_csv(os.path.join("data", "biz_csv", 'business.csv'))
print(business_df)
'''business_csv_file = open(os.path.join("..","data","biz_csv","business.csv"),"r", encoding="utf8")
business_csv = csv.reader(business_csv_file)
categories = list()
first_line = 1
    if first_line > 1:
        categories.append([float(line[1]), float(line[2])])
    first_line += 1'''

rkfold = RepeatedKFold(n_splits=5, n_repeats=1, random_state=1)
tem = list()
something = rkfold.split(business_df)
for train,test in rkfold.split(business_df):
    print(len(train), len(test))
    temp =business_df.values[train]
    temp_ = business_df.values[test]
    print(temp)
    tem.append(temp)
#print(tem)
#print(tem)



