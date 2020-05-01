#!/usr/bin/env python
# coding: utf-8

# # Oakland Crime Statistics 2011 to 2016 

# ## 数据预处理
# 首先对数据进行预处理，将时间的单词

# In[26]:


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


# In[34]:


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


# In[37]:



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

# In[16]:


crimes_2012 = pd.read_csv("./oakland-crime-statistics-2011-to-2016/records-for-2012.csv")
crimes_2012.dropna(thresh=9, inplace=True)
crimes_2012 = prep_data(crimes_2012)
crimes_2012.rename(index=str, columns={"Location ": "address"}, inplace=True)
crimes_2012["Area Id"] = crimes_2013["Area Id"].astype(int)
crimes_2012["Priority"].replace(0.0, 1.0, inplace=True)
crimes_2012["Priority"] = crimes_2013["Priority"].astype(int)
crimes_2012.head(2)


# - 2013年数据

# In[14]:


crimes_2013 = pd.read_csv("./oakland-crime-statistics-2011-to-2016/records-for-2013.csv")
crimes_2013.dropna(thresh=9, inplace=True)
crimes_2013 = prep_data(crimes_2013)
crimes_2013.rename(index=str, columns={"Location ": "address"}, inplace=True)
crimes_2013["Area Id"] = crimes_2013["Area Id"].astype(int)
crimes_2013["Priority"].replace(0.0, 1.0, inplace=True)
crimes_2013["Priority"] = crimes_2013["Priority"].astype(int)
crimes_2013.head(2)


# - 2014年数据

# In[17]:


crimes_2014 = pd.read_csv("./oakland-crime-statistics-2011-to-2016/records-for-2014.csv")
crimes_2014.dropna(thresh=9, inplace=True)
crimes_2014 = prep_data(crimes_2014)
crimes_2014.rename(index=str, columns={"Location ": "address"}, inplace=True)
crimes_2014["Area Id"] = crimes_2013["Area Id"].astype(int)
crimes_2014["Priority"].replace(0.0, 1.0, inplace=True)
crimes_2014["Priority"] = crimes_2013["Priority"].astype(int)
crimes_2014.head(2)


# - 2015年数据

# In[18]:


crimes_2015 = pd.read_csv("./oakland-crime-statistics-2011-to-2016/records-for-2015.csv")
crimes_2015.dropna(thresh=9, inplace=True)
crimes_2015 = prep_data(crimes_2015)
crimes_2015.rename(index=str, columns={"Location ": "address"}, inplace=True)
crimes_2015["Area Id"] = crimes_2013["Area Id"].astype(int)
crimes_2015["Priority"].replace(0.0, 1.0, inplace=True)
crimes_2015["Priority"] = crimes_2013["Priority"].astype(int)
crimes_2015.head(2)


# - 2016年数据

# In[24]:


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


# In[44]:


fig, ax = plt.subplots(nrows=2, ncols=3)
plt.subplots_adjust(left=0, right=2.5, top=3, bottom=1)
i = 0
for row in ax:
    for col in row:
        col.set_title(str(2011 + i))
        temp = crimes_list[i].groupby(by=["Beat", "Priority"],sort=True, as_index=False).count().rename(index=str, columns={"Create Time": "Count"})[["Beat", "Priority", "Count", "time_of_the_day"]]
        beats_prio_1 = list(temp[temp["Priority"] == 1].nlargest(5, "Count")["Beat"].values)
        beats_prio_2 = list(temp[temp["Priority"] == 2].nlargest(5, "Count")["Beat"].values)
        sns.countplot(data=crimes_list[i][crimes_list[i]["Beat"].isin(beats_prio_1 + beats_prio_2)], x="Beat", hue="time_of_the_day",palette="Set1", ax=col)
        i += 1


# In[47]:


for i, x in enumerate(crimes_list):
    x["Year"] = 2011 + i
combined = crimes_2011
for x in range(1,len(crimes_list)):
    combined = combined.append(crimes_list[x], ignore_index=True)
combined.tail(5)


# In[48]:


temp = combined.groupby(by=["Year", "Priority"]).mean()
prio_1 = temp.loc[list(zip(range(2011,2017),[1.0] * 6))]["time_between_creation_and_closed_seconds"]
prio_2 = temp.loc[list(zip(range(2011,2017),[2.0] * 6))]["time_between_creation_and_closed_seconds"]
plt.plot(range(2011, 2017),prio_1, marker='o', markerfacecolor='black', markersize=8, color='skyblue', linewidth=2, label="Avg Closing Time Priority 1")
plt.plot(range(2011, 2017), prio_2, marker='*',color="red", markersize=10, markerfacecolor='black', linewidth=2, label="Avg Closing Time Priority 2")
plt.legend()


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


# ### 每年犯罪持续时间盒图

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

# In[29]:


fig, ax = plt.subplots(nrows=2, ncols=3)
plt.subplots_adjust(left=0, right=2.5, top=3, bottom=1)
i = 0
nlargest = [set(get_nlargest_incident_id(10, x)) for x in crimes_list]
print("From 2011 to 2016 Top 10 Common Incident Types are: {}".format(str(set.intersection(*nlargest))))
for row in ax:
    for col in row:
        col.set_title(str(2011 + i))
        sns.countplot(data=crimes_list[i].loc[crimes_list[i]['Incident Type Id'].isin(nlargest[i])], x="Incident Type Id", hue="Priority", palette="Set1", ax=col)
        i += 1


# In[36]:


fig, ax = plt.subplots(nrows=2, ncols=3)
plt.subplots_adjust(left=0, right=2.5, top=3, bottom=1)
i = 0
area_nlargest = [set(get_nlargest_area_id(10, x)) for x in crimes_list]
print("From 2011 to 2016 Top 10 Common Area Id are: {}".format(str(set.intersection(*area_nlargest))))
for row in ax:
    for col in row:
        col.set_title(str(2011 + i))
        sns.countplot(data=crimes_list[i].loc[crimes_list[i]['Area Id'].isin(area_nlargest[i])], x="Area Id", hue="Priority", palette="Set1", ax=col)
        i += 1


# In[41]:


fig, ax = plt.subplots(nrows=6, ncols=3)
plt.subplots_adjust(left=0, right=3, top=12, bottom=0)
i_list = 0
for i, row in enumerate(ax):
    for j, col in enumerate(row):
        year_string = str(2011 + i)
        
        if j == 1:
            month_or_day = 'Day of The Month'
            title = year_string+ '\n' + month_or_day +'\n Crime Count'
            col.set_title(title)
            col.set_xticklabels(col.get_xticklabels(), rotation=90)
            sns.countplot(data=crimes_list[i_list], x="day_of_the_month" ,palette="Set1", ax=col)
        elif j == 2:
            month_or_day = 'Month of The Year'
            title = year_string + '\n' + month_or_day +'\n Crime Count'
            col.set_title(title)
            sns.countplot(data=crimes_list[i_list], x="month_of_the_year",palette="Set1", ax=col)
        else:
            month_or_day = 'Day of The Week'
            title = year_string+ '\n' + month_or_day +'\n Crime Count'
            col.set_title(title)
            sns.countplot(data=crimes_list[i_list], x="day_of_the_week" ,palette="Set1", ax=col)
            
    i_list += 1

