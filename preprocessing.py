#Data Preprocessing
import os
import pandas as pd
import re


def clean_data(yourdata):

    #Drop '0' rating
    yourdata = yourdata[yourdata.rating != 0]
    print("After cleaning 0s, the number of reviews are ", len(yourdata))
    # Count the amount of each rating. 
    print(yourdata['rating'].value_counts())

    # 
    yourdata['review']=yourdata['review'].str.replace("\n", "")
    yourdata['review']=yourdata['review'].apply(lambda x : re.sub(r'\([^)]*\)'," ", x.lower()))
    yourdata['review']=yourdata['review'].apply(lambda x : re.sub("[^a-zA-Z]"," ", x))
    #print(yourdata['review'][0])


    return yourdata

def read_csv(file_name):

    #Drop rows if there is none value. 
    yourdata=pd.read_csv(file_name).dropna(axis=0)
    #Check your data is correctly read. 
    
    print ('head: ', yourdata.head(), 'first review', yourdata['review'][0])

    yourdata = clean_data(yourdata)
    print('---Data imported.---- ')
    
    return yourdata






