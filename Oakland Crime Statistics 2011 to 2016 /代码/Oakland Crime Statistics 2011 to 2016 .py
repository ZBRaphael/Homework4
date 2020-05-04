#!/usr/bin/env python
# coding: utf-8

# # Oakland Crime Statistics 2011 to 2016 

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import json
import ast
import datetime as dt
from dateutil import tz
from dateutil import parser
import matplotlib.pyplot as plt
import time

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print('\n'.join(os.listdir("./oakland-crime-statistics-2011-to-2016/")))


# ## 数据预处理
# 
# 首先对数据进行预处理，将时间的换算成秒，将时间划分成几个时间段，并且将数据中的空值直接去除。

# In[2]:


intervals = (
    ('weeks', 604800),  # 60 * 60 * 24 * 7
    ('days', 86400),    # 60 * 60 * 24
    ('hours', 3600),    # 60 * 60
    ('minutes', 60),
    ('seconds', 1),
    )


def get_nlargest_incident_id(n, df):
    return df.groupby(by="Incident Type Id",sort=True, as_index=False).count().nlargest(n, 'Create Time')["Incident Type Id"].values
def get_nlargest_area_id(n, df):
    return df.groupby(by="Area Id",sort=True, as_index=False).count().nlargest(n, 'Create Time')["Area Id"].values
def display_time(seconds, granularity=10):
    result = []

    for name, count in intervals:
        value = seconds // count
        if value:
            seconds -= value * count
            if value == 1:
                name = name.rstrip('s')
            result.append("{} {}".format(value, name))
    return ', '.join(result[:granularity])

def map_x(x):
    if x.hour < 6:
        return "00AM-6AM"
    if x.hour < 12 and x.hour > 6:
        return "6AM-12PM"
    if x.hour >= 12 and x.hour<18:
        return "12PM-6PM"
    if x.hour > 18:
        return "6PM-00AM"
    
def prep_data(df):
    df['Create Time'] = df['Create Time'].fillna(df['Closed Time'])
    df['Closed Time'] = df['Closed Time'].fillna(df['Create Time'])
    df["time_between_creation_and_closed_seconds"] = df.apply(lambda x: abs((parser.parse(x["Closed Time"]) - parser.parse(x["Create Time"])).seconds), axis=1)
    df["time_of_the_day"] = df["Create Time"].map(lambda x:map_x(parser.parse(x)))
    df.replace(r'', np.nan, regex=True, inplace=True)
    df["Area Id"].fillna(-1, inplace=True)
    df["Beat"].fillna("Unknown", inplace=True)
    df["Priority"].fillna("-1", inplace=True)
    df["Priority"].astype(int)
    df.drop(["Agency", "Event Number"], inplace=True, axis=1)
    df["day_of_the_month"] = df["Create Time"].apply(lambda x: parser.parse(x).day)
    df["day_of_the_week"] = df["day_of_the_month"].apply(lambda x: (x % 7) + 1)
    df["month_of_the_year"] = df["Create Time"].apply(lambda x: parser.parse(x).month)
    return df


# In[3]:



# 五数分布


def fiveNumber(nums):
        # 五数概括 Minimum（最小值）、Q1、Median（中位数、）、Q3、Maximum（最大值）
    Minimum = min(nums)
    Maximum = max(nums)
    Q1 = np.percentile(nums, 25)
    Median = np.median(nums)
    Q3 = np.percentile(nums, 75)

    IQR = Q3-Q1
    lower_limit = Q1-1.5*IQR  # 下限值
    upper_limit = Q3+1.5*IQR  # 上限值

    return Minimum, Q1, Median, Q3, Maximum, lower_limit, upper_limit


def printfiveNumber(fivenumber):
    print("+++++++++++++++")
    print(f"Min = {fivenumber[0]}")
    print(f"Q1  = {fivenumber[1]}")
    print(f"Median = {fivenumber[2]}")
    print(f"Q3 = {fivenumber[3]}")
    print(f"Max = {fivenumber[4]}")
    print(f"lower_limit = {fivenumber[5]}")
    print(f"upper_limit = {fivenumber[6]}")
    print("+++++++++++++++")


# ## 导入数据
# 

# - 2011年数据

# In[4]:


crimes_2011 = pd.read_csv("./oakland-crime-statistics-2011-to-2016/records-for-2011.csv")
crimes_2011.drop(index=[180015], inplace=True)
crimes_2011 = prep_data(crimes_2011)
crimes_2011.rename(index=str, columns={"Location": "address"}, inplace=True)
crimes_2011["Priority"].replace(0.0, 1.0, inplace=True)
crimes_2011["Priority"] = crimes_2011["Priority"].astype(int)
crimes_2011.head(2)


# - 2012年数据

# In[6]:


crimes_2012 = pd.read_csv("./oakland-crime-statistics-2011-to-2016/records-for-2012.csv")
crimes_2012.dropna(thresh=9, inplace=True)
crimes_2012 = prep_data(crimes_2012)
crimes_2012.rename(index=str, columns={"Location ": "address"}, inplace=True)
crimes_2012["Area Id"] = crimes_2012["Area Id"].astype(int)
crimes_2012["Priority"].replace(0.0, 1.0, inplace=True)
crimes_2012["Priority"] = crimes_2012["Priority"].astype(int)
crimes_2012.head(2)


# - 2013年数据

# In[7]:


crimes_2013 = pd.read_csv("./oakland-crime-statistics-2011-to-2016/records-for-2013.csv")
crimes_2013.dropna(thresh=9, inplace=True)
crimes_2013 = prep_data(crimes_2013)
crimes_2013.rename(index=str, columns={"Location ": "address"}, inplace=True)
crimes_2013["Area Id"] = crimes_2013["Area Id"].astype(int)
crimes_2013["Priority"].replace(0.0, 1.0, inplace=True)
crimes_2013["Priority"] = crimes_2013["Priority"].astype(int)
crimes_2013.head(2)


# - 2014年数据

# In[8]:


crimes_2014 = pd.read_csv("./oakland-crime-statistics-2011-to-2016/records-for-2014.csv")
crimes_2014.dropna(thresh=9, inplace=True)
crimes_2014 = prep_data(crimes_2014)
crimes_2014.rename(index=str, columns={"Location ": "address"}, inplace=True)
crimes_2014["Area Id"] = crimes_2013["Area Id"].astype(int)
crimes_2014["Priority"].replace(0.0, 1.0, inplace=True)
crimes_2014["Priority"] = crimes_2013["Priority"].astype(int)
crimes_2014.head(2)


# - 2015年数据

# In[9]:


crimes_2015 = pd.read_csv("./oakland-crime-statistics-2011-to-2016/records-for-2015.csv")
crimes_2015.dropna(thresh=9, inplace=True)
crimes_2015 = prep_data(crimes_2015)
crimes_2015.rename(index=str, columns={"Location ": "address"}, inplace=True)
crimes_2015["Area Id"] = crimes_2013["Area Id"].astype(int)
crimes_2015["Priority"].replace(0.0, 1.0, inplace=True)
crimes_2015["Priority"] = crimes_2013["Priority"].astype(int)
crimes_2015.head(2)


# - 2016年数据

# In[10]:


crimes_2016 = pd.read_csv("./oakland-crime-statistics-2011-to-2016/records-for-2016.csv")
crimes_2016.dropna(thresh=9, inplace=True)
crimes_2016 = prep_data(crimes_2016)
crimes_2016.rename(index=str, columns={"Location": "address"}, inplace=True)
crimes_2016["Priority"] = crimes_2016["Priority"].astype(int)
crimes_2016.head(2)


# ## 数据可视化和摘要
# 
# 

# ### 犯罪持续时间五数分布

# In[40]:


print("五数分析")
for i, crime_year in enumerate(crimes_list):
    five = fiveNumber(crime_year[crime_year["Priority"] == 1]["time_between_creation_and_closed_seconds"])
    print(str(2011+i)+"time_between_creation_and_closed_seconds")
    printfiveNumber(five)


# ### 犯罪处理时间随时间的变化

# In[100]:


def box_plot(all_data):
#     all_data = np.array(all_data)
    fig = plt.figure(figsize=(16,8))

    plt.boxplot(all_data,notch=False, sym='o',vert=True)   # vertical box aligmnent  # vertical box aligmnent
    year_list = []
    for i in range(2011,2017):
        year_list.append(str(i))
    plt.xticks([i for i in range(1,7)],year_list)
#     plt.xlabel('measurement x')
    t = plt.title('Box plot')
    plt.show()


# ### 每年犯罪持续时间盒图

# In[101]:


box = []
for crime_year in crimes_list:
    plt.subplots_adjust(left=0, right=2.5, top=3, bottom=1)
    box.append(crime_year["time_between_creation_and_closed_seconds"])
# box = box.transpose()
print(np.shape(box))
#         plt.boxplot(box,notch=False, sym='o',vert=True)   # vertical box aligmnent  # vertical box aligmnent
#         plot.show()
    
box_plot(box)


# ### beats的频数统计

# In[43]:



fig, ax = plt.subplots(nrows=2, ncols=3)
plt.subplots_adjust(left=0, right=2.5, top=3, bottom=1)
i = 0
for row in ax:
    for col in row:
        col.set_title(str(2011 + i))
        #sns.countplot(data=crimes_list[i].loc[crimes_list[i]['Incident Type Id'].isin(nlargest[i])], x="Incident Type Id", hue="Priority", palette="Set1", ax=col)
        temp = crimes_list[i].groupby(by=["Beat", "Priority"],sort=True, as_index=False).count().rename(index=str, columns={"Create Time": "Count"})[["Beat", "Priority", "Count", "time_of_the_day"]]
        beats_prio_1 = list(temp[temp["Priority"] == 1].nlargest(5, "Count")["Beat"].values)
        beats_prio_2 = list(temp[temp["Priority"] == 2].nlargest(5, "Count")["Beat"].values)
        print("Year " + str(2011 +i ) +":\n")
        print("The Beats With the Most Reports (Priority 1, Decending Order): {} \nThe Beats With the Most Reports (Priority 2, Decending Order): {} \nUnique Beats: {}".format(str(beats_prio_1), str(beats_prio_2), str(list(set(beats_prio_1)|set(beats_prio_2)))))
        print("Common Beats: {}".format(str(list(set(beats_prio_1) & set(beats_prio_2)))))
        sns.barplot(data=temp[temp["Beat"].isin(beats_prio_1 + beats_prio_2)], x="Beat", y="Count", hue="Priority",palette="Set1", ax=col)
        print("=======================================================================================\n")
        i += 1


# ### 每年的报警数量

# In[27]:


fig, ax = plt.subplots(nrows=2, ncols=3)
plt.subplots_adjust(left=0, right=2.5, top=3, bottom=1)
crimes_list = [crimes_2011, crimes_2012, crimes_2013, crimes_2014, crimes_2015, crimes_2016]
i = 0
for row in ax:
    for col in row:
        col.set_title(str(2011 + i))
        sns.countplot(data=crimes_list[i], x="Priority", ax=col, palette="Set1")
        i+=1


# ### 每年报警的时间段分布

# In[28]:


fig, ax = plt.subplots(nrows=2, ncols=3)
plt.subplots_adjust(left=0, right=2.5, top=3, bottom=1)
i = 0
for row in ax:
    for col in row:
        col.set_title(str(2011 + i))
        sns.countplot(data=crimes_list[i], x="Priority", hue="time_of_the_day", palette="Set1", ax=col)
        i+=1


# 从上述的图中可以看出优先级为1的案件少，优先级为2的案件多，案件的高发时间段为12PM-6PM

# ## 缺失数据的处理

# In[64]:


crimes_2011_miss = pd.read_csv("./oakland-crime-statistics-2011-to-2016/records-for-2011.csv", keep_default_na=False)
crimes_2011_miss.head(2)


# In[19]:


crimes_2012_miss = pd.read_csv("./oakland-crime-statistics-2011-to-2016/records-for-2012.csv", keep_default_na=False)
crimes_2012_miss.head(2)


# In[20]:


crimes_2013_miss = pd.read_csv("./oakland-crime-statistics-2011-to-2016/records-for-2013.csv", keep_default_na=False)
crimes_2013_miss.head(2)


# In[21]:


crimes_2014_miss = pd.read_csv("./oakland-crime-statistics-2011-to-2016/records-for-2014.csv", keep_default_na=False)
crimes_2014_miss.head(2)


# In[22]:


crimes_2015_miss = pd.read_csv("./oakland-crime-statistics-2011-to-2016/records-for-2015.csv", keep_default_na=False)
crimes_2015_miss.head(2)


# In[23]:


crimes_2016_miss = pd.read_csv("./oakland-crime-statistics-2011-to-2016/records-for-2016.csv", keep_default_na=False)
crimes_2016_miss.head(2)


# In[116]:


def loaddata():
    crimes_2011_miss = pd.read_csv("./oakland-crime-statistics-2011-to-2016/records-for-2011.csv", keep_default_na=False)

    crimes_2012_miss = pd.read_csv("./oakland-crime-statistics-2011-to-2016/records-for-2012.csv", keep_default_na=False)

    crimes_2013_miss = pd.read_csv("./oakland-crime-statistics-2011-to-2016/records-for-2013.csv", keep_default_na=False)

    crimes_2014_miss = pd.read_csv("./oakland-crime-statistics-2011-to-2016/records-for-2014.csv", keep_default_na=False)

    crimes_2015_miss = pd.read_csv("./oakland-crime-statistics-2011-to-2016/records-for-2015.csv", keep_default_na=False)

    crimes_2016_miss = pd.read_csv("./oakland-crime-statistics-2011-to-2016/records-for-2016.csv", keep_default_na=False)

    miss_list = [crimes_2011_miss,crimes_2012_miss,crimes_2013_miss,crimes_2014_miss,crimes_2015_miss,crimes_2016_miss]
    
    return miss_list
def countbeat():
    fig, ax = plt.subplots(nrows=2, ncols=3)
    plt.subplots_adjust(left=0, right=2.5, top=3, bottom=1)
    i = 0
    for row in ax:
        for col in row:
            col.set_title(str(2011 + i))
            sns.countplot(x="Beat", data=miss_list[i],ax=col, palette="Set1") 
            i+=1


# In[27]:


def count_ana(pd,col):
    countana = 0
    for i in pd[col]:
        if i == '':
            countana += 1
    return countana    


# In[58]:


miss_list = [crimes_2011_miss,crimes_2012_miss,crimes_2013_miss,crimes_2014_miss,crimes_2015_miss,crimes_2016_miss]
for dataset in miss_list:
    count = count_ana(dataset,"Priority")
    print(count)
# count_priority


# Priority字段几乎无缺失值

# In[59]:


for dataset in miss_list:
    count = count_ana(dataset,"Incident Type Id")
    print(count)


# Incident Type Id字段也无缺失

# In[60]:


for dataset in miss_list:
    count = count_ana(dataset,"Beat")
    print(count)


# 我们选择对“Beat”进行缺失值的填充
# ### 直接将缺失的数值剔除
# 在数据上部分的数据预处理过程中，已经将缺失的数据删除

# In[61]:


crimes_list = [crimes_2011, crimes_2012, crimes_2013, crimes_2014, crimes_2015, crimes_2016]

for dataset in crimes_list:
    count = count_ana(dataset,"Beat")
    print(count)


# ### 用频数最大的值进行填充

# In[55]:


maxcount = []
for pd in crimes_list:
    count = pd["Beat"].value_counts()
#     maxcount.append(count)
    print(count.iloc[:1])
# print(maxcount)


# 我们发现每年的数据中出现频率最高的是04X，所以用04X进行填充
# 

# In[117]:


miss_list = loaddata()
print(len(miss_list))
for dataset in miss_list:
    count = count_ana(dataset,"Beat")
    print(count)


# In[118]:


countbeat()


# In[119]:


for dataset in miss_list:
    for i in range(len(dataset["Beat"])):
        if dataset["Beat"][i] == '':
            dataset["Beat"][i] = "04X"


# In[120]:


for dataset in miss_list:
    count = count_ana(dataset,"Beat")
    print(count)


# In[121]:


countbeat()


# ### 使用与空缺值相邻的值进行填充

# In[122]:


miss_list = loaddata()
for dataset in miss_list:
    count = count_ana(dataset,"Beat")
    print(count)


# In[124]:


for dataset in miss_list:
    for i in range(len(dataset["Beat"])):
        if dataset["Beat"][i] == '':
            for j in range(5):
                if dataset["Beat"][i-j] != '':
                    dataset["Beat"][i] = dataset["Beat"][i-j]


# In[125]:


for dataset in miss_list:
    count = count_ana(dataset,"Beat")
    print(count)


# In[126]:


countbeat()


# ### 通过数据对象之间的相似性来填补缺失值

# In[127]:


miss_list = loaddata()
for dataset in miss_list:
    count = count_ana(dataset,"Beat")
    print(count)


# In[130]:


for dataset in miss_list: 
    beatdataset = dataset["Beat"]
    typedataset = dataset["Incident Type Id"]
    dir_type_beat = {}
    for i in range(len(beatdataset)):
        if beatdataset[i] != '':
            dir_type_beat[typedataset[i]] =  beatdataset[i]
    for i in range(len(beatdataset)):
        if beatdataset[i] == '':
            if typedataset[i] not in dir_type_beat.keys():
                beatdataset[i] = "04X"
            else:
                beatdataset[i] = dir_type_beat[typedataset[i]]


# In[131]:


for dataset in miss_list:
    count = count_ana(dataset,"Beat")
    print(count)


# In[132]:


countbeat()


# 由于缺失的值只占很少的部分所以对于频数分布来说并没有什么很大的变化
