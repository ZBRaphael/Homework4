#!/usr/bin/env python
# coding: utf-8

# # 频繁模式与关联规则挖掘

# In[1]:


# Import modules
import numpy as np
import pandas as pd
import seaborn as sns
import bokeh as bk
from bokeh.io import output_notebook, show
output_notebook()
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[2]:


# Ignore irrelevant warnings 
import warnings
warnings.filterwarnings('ignore')


# In[3]:


# 设置整体绘图打印样式
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")
sns.set() 
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.labelpad'] = 10
sns.set_style("darkgrid")
# sns.set_palette("Reds")
# sns.set_context("poster", font_scale=1.0)


# In[97]:


WinerDF = pd.read_csv("./wine-reviews/winemag-data-130k-v2.csv")


# In[171]:


WineData = pd.DataFrame(WinerDF,columns = ["country","points","price"])


# In[172]:


WineData.info()


# In[173]:


WineData.head()


# ## 分析规则
# 为了使得该数据集适用于关联规则挖掘，我们对数据进行了一下简单的重构。对于数值属性“points”,“price”进行离散化处理。根据数值的大小，将他们分别分为四个等级，表示评分的高低及价格的高低。并将离散后的数据转换为适用于Apriori算法处理的transactions类型。

# In[174]:


WineData['points'].describe()


# In[175]:


WineData['price'].describe()


# In[176]:


for i in range(len(WineData['points'])):
#     print(type(i))
    if WineData['points'][i] >= 95:
        WineData['points'][i] = "po1st"
    elif WineData['points'][i] >= 90:
        WineData['points'][i] = "po2nd"
    elif WineData['points'][i] >= 85:
        WineData['points'][i] = "po3rd"
    else:
        WineData['points'][i] = "po4th"


# In[177]:


for i in range(len(WineData['price'])):
#     print(type(i))
    if WineData['price'][i] >= 200:
        WineData['price'][i] = "pr1st"
    elif WineData['price'][i] >= 100:
        WineData['price'][i] = "pr2nd"
    elif WineData['price'][i] >= 50:
        WineData['price'][i] = "pr3rd"
    else:
        WineData['price'][i] = "pr4th"


# In[179]:


WineData.head()


# In[180]:


WineData["country"].value_counts().plot.bar()


# In[181]:


WineData["points"].value_counts().plot.bar()


# In[182]:


WineData["price"].value_counts().plot.bar()


# 将Pandas DataFrame转换为列表

# In[183]:


records = WineData.to_records(index=False)
result = list(records)


# In[184]:


result[0:10]


# In[185]:


from efficient_apriori import apriori


# In[198]:


rules = apriori(result, min_support=0.01, min_confidence=0.7)
print(rules)  


# 根据关联结果中的提升度(life)进行降序排序。
# 上面满足支持度阈值和置信度阈值的规则存在冗余规则，冗余规则的定义是：如果rules2的lhs和rhs是包含于rules1的，而且rules2的lift小于或者等于rules1，则称rules2是rules1的冗余规则。下面对冗余规则进行删除，最终关联规则精简到11条。

# In[202]:


itemsets, rules = apriori(result, min_support=0.01, min_confidence=0.7)

# Print out every rule with 2 items on the left hand side,
# 1 item on the right hand side, sorted by lift
rules_rhs = filter(lambda rule:  len(rule.lhs) == 2 and len(rule.rhs) == 1, rules)
i = 0
for rule in sorted(rules_rhs, key=lambda rule: rule.lift,reverse = True):
    print(str(i)+":",end="")
    print(rule)  # Prints the rule and its confidence, support, lift, ...
    i+=1


# ## 可视化

# In[252]:


rules_rhs = filter(lambda rule:  len(rule.lhs) == 2 and len(rule.rhs) == 1, rules)
support = []
confidence = []
lift = []
conviction = []
for rule in sorted(rules_rhs, key=lambda rule: rule.lift,reverse = True):
    support.append(rule.support)
    confidence.append(rule.confidence)
    lift.append((rule.lift-0.95)*5000)
    conviction.append(rule.conviction*100)


# In[249]:


import random
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.subplot()
ax.scatter(support, confidence, s=lift, alpha=0.5)  # 绘制散点图，
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()


# 圆圈的面积代表lift的大小

# In[253]:


fig = plt.figure()
ax = plt.subplot()
ax.scatter(support, confidence, s=conviction, alpha=0.5)  # 绘制散点图，
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()


# 圆圈的面积代表确信度（conviction）的大小
