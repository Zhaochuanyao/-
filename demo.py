import pandas as pd
import csv
import matplotlib.pyplot as plt
data = pd.read_csv(r'C:\Users\zcy\PycharmProjects\pythonProject4\data\bankriskinfo.csv',encoding='utf-8')
#print(data)

#将列表转换为DataFrame
data = pd.DataFrame(data)
print(data[:1])
data_5 = data.head(5)
print(data_5)
print(data.info())
# 使用describe()函数查看数据整体的基本统计信息
data_des = data.describe(include='all')
print(data_des)


fig = plt.figure(figsize=(8,6))
# 绘制柱状图，查看违约关系的取值分布情况
data['Default1'].value_counts(dropna=False).plot(kind='bar',rot=40) #不去除nan值,x轴标签旋转40度

# 在柱形上方显示计数
counts = data['Default1'].value_counts(dropna=False).values
for index, item in zip([0,1,2], counts):
    plt.text(index, item, item, ha="center", va= "bottom", fontsize=12)

# 设置柱形名称
plt.xticks([0,1,2],['未违约','违约','NaN'])
# 设置x、y轴标签
plt.xlabel("是否违约")
plt.ylabel("客户数量")
# 设置标题以及字体大小
plt.title("违约与未违约数量分布图",size=13)
# 设置中文显示
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['font.family']=['sans-serif']
#plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(16, 6))
# 对CityId列的类别设定顺序
data['CityId'] = data['CityId'].astype('category')
data['CityId'] = data['CityId'].cat.set_categories(['一线城市', '二线城市', '其它'], ordered=True)

# 绘制柱状图，查看不同城市级别在不同是否违约的取值分布情况
sns.countplot(x='CityId', hue='Default1', data=data, ax=ax1)

# 将具体的计数值显示在柱形上方
counts = data['Default1'].groupby(data['CityId']).value_counts().values
count1 = counts[[0, 2, 4]]
count2 = counts[[1, 3, 5]]
for index, item1, item2 in zip([0, 1, 2], count1, count2):
    ax1.text(index - 0.2, item1 + 0.05, '%.0f' % item1, ha="center", va="bottom", fontsize=12)
    ax1.text(index + 0.2, item2 + 0.05, '%.0f' % item2, ha="center", va="bottom", fontsize=12)

# 绘制柱状图查看违约率分布
cityid_rate = data.groupby('CityId')['Default1'].sum() / data.groupby('CityId')['Default1'].count()
sns.barplot(x=[0, 1, 2], y=cityid_rate, ax=ax2)

# 将具体的计数值显示在柱形上方
for index, item in zip([0, 1, 2], cityid_rate):
    ax2.text(index, item, '%.3f' % item, ha="center", va="bottom", fontsize=12)

# 设置柱形名称
ax1.set_xticklabels(['一线城市', '二线城市', '其它'])
ax2.set_xticklabels(['一线城市', '二线城市', '其它'])

# 设置图例名称
ax1.legend(['未违约', '违约'])

# 设置标题以及字体大小
ax1.set_title("不同城市级别下不同违约情况数量分布柱状图", size=13)
ax2.set_title("不同城市级别违约率分布柱状图", size=13)

# 设置x,y轴标签
ax1.set_xlabel("CityId")
ax1.set_ylabel("客户人数")
ax2.set_xlabel("CityId")
ax2.set_ylabel("违约率")

# 显示汉语标注
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family'] = ['sans-serif']
#plt.show()

fig,[ax1,ax2] = plt.subplots(1,2,figsize=(16,6))

# 对education列的类别设定顺序
data['education'] = data['education'].astype('category')
data['education'] = data['education'].cat.set_categories(['小学', '初中', '高中', '本科以上'],ordered=True)

# 绘制柱状图，查看不同文化程度(education)在不同是否违约(Default)的取值分布情况
sns.countplot(x='education', hue='Default1', data=data, ax=ax1)

# 将具体的计数值显示在柱形上方
counts=data['Default1'].groupby(data['education']).value_counts().values
count1 = counts[[0, 2, 4,6]]
count2 = counts[[1, 3, 5,7]]
for index, item1, item2 in zip([0,1,2,3], count1, count2):
    ax1.text(index-0.2, item1 + 0.05, '%.0f' % item1, ha="center", va= "bottom",fontsize=12)
    ax1.text(index+0.2, item2 + 0.05, '%.0f' % item2, ha="center", va= "bottom",fontsize=12)

# 绘制柱状图查看违约率分布
education_rate = data.groupby('education')['Default1'].sum() / data.groupby('education')['Default1'].count()
sns.barplot(x=[0,1,2,3],y=education_rate.values,ax=ax2)

# 将具体的计数值显示在柱形上方
for index, item in zip([0,1,2,3], education_rate):
     ax2.text(index, item, '%.2f' % item, ha="center", va= "bottom",fontsize=12)

        # 设置柱形名称
ax1.set_xticklabels(['小学', '初中', '高中', '本科以上'])
ax2.set_xticklabels(['小学', '初中', '高中', '本科以上'])

# 设置图例名称
ax1.legend(['未违约','违约'])

# 设置标题以及字体大小
ax1.set_title("不同文化程度下不同违约情况数量分布柱状图",size=13)
ax2.set_title("不同文化程度下违约率分布柱状图",size=13)

# 设置x,y轴标签
ax1.set_xlabel("education")
ax1.set_ylabel("客户人数")
ax2.set_xlabel("education")
ax2.set_ylabel("违约率")

#显示汉语标注
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['font.family']=['sans-serif']
#plt.show()


fig,[ax1,ax2] = plt.subplots(1,2,figsize=(16,6))

# 对threeVerify列的类别设定顺序
data['threeVerify'] = data['threeVerify'].astype('category')
data['threeVerify'] = data['threeVerify'].cat.set_categories(['一致','不一致'],ordered=True)

# 绘制柱状图，查看不同三要素验证情况(threeVerify)在不同是否违约(Default)的取值分布情况
sns.countplot(x='threeVerify', hue='Default1', data=data, ax=ax1)

# 将具体的计数值显示在柱形上方
counts=data['Default1'].groupby(data['threeVerify']).value_counts().values
count1 = counts[[0, 2]]
count2 = counts[[1, 3]]
for index, item1, item2 in zip([0,1,2,3], count1, count2):
    ax1.text(index-0.2, item1 + 0.05, '%.0f' % item1, ha="center", va= "bottom",fontsize=12)
    ax1.text(index+0.2, item2 + 0.05, '%.0f' % item2, ha="center", va= "bottom",fontsize=12)

# 绘制柱状图查看违约率分布
threeVerify_rate = data.groupby('threeVerify')['Default1'].sum() / data.groupby('threeVerify')['Default1'].count()
sns.barplot(x=[0,1],y=threeVerify_rate.values,ax=ax2)

# 将具体的计数值显示在柱形上方
for index, item in zip([0,1], threeVerify_rate):
     ax2.text(index, item, '%.2f' % item, ha="center", va= "bottom",fontsize=12)

# 设置柱形名称
ax1.set_xticklabels(['一致','不一致'])
ax2.set_xticklabels(['一致','不一致'])

# 设置图例名称
ax1.legend(['未违约','违约'])

# 设置标题以及字体大小
ax1.set_title("不同三要素验证下不同违约情况数量分布柱状图",size=13)
ax2.set_title("不同三要素验证下违约率分布柱状图",size=13)

# 设置x,y轴标签
ax1.set_xlabel("threeVerify")
ax1.set_ylabel("客户人数")
ax2.set_xlabel("threeVerify")
ax2.set_ylabel("违约率")

#显示汉语标注
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['font.family']=['sans-serif']
plt.show()

fig,[ax1,ax2] = plt.subplots(1,2,figsize=(16,6))
# 对maritalStatus列的类别设定顺序
data['maritalStatus'] = data['maritalStatus'].astype('category')
data['maritalStatus'] = data['maritalStatus'].cat.set_categories(['未婚','已婚'],ordered=True)
# 绘制柱状图，查看不同婚姻状况在不同违约情况的取值分布
sns.countplot(x='maritalStatus', hue='Default1', data=data, ax=ax1)
# 将具体的计数值显示在柱形上方
counts=data['Default1'].groupby(data['maritalStatus']).value_counts().values
count1 = counts[[0, 2]]
count2 = counts[[1, 3]]
for index, item1, item2 in zip([0,1,2,3], count1, count2):
    ax1.text(index-0.2, item1 + 0.05, '%.0f' % item1, ha="center", va= "bottom",fontsize=12)
    ax1.text(index+0.2, item2 + 0.05, '%.0f' % item2, ha="center", va= "bottom",fontsize=12)
# 绘制柱状图查看违约率分布
maritalStatus_rate = data.groupby('maritalStatus')['Default1'].sum() / data.groupby('maritalStatus')['Default1'].count()
sns.barplot(x=[0,1],y=maritalStatus_rate.values,ax=ax2)
# 将具体的计数值显示在柱形上方
for index, item in zip([0,1], maritalStatus_rate):
     ax2.text(index, item, '%.2f' % item, ha="center", va= "bottom",fontsize=12)
# 设置柱形名称
ax1.set_xticklabels(['未婚','已婚'])
ax2.set_xticklabels(['未婚','已婚'])
# 设置图例名称
ax1.legend(['未违约','违约'])
# 设置标题以及字体大小
ax1.set_title("不同婚姻状况下不同违约情况数量分布柱状图",size=13)
ax2.set_title("不同婚姻状况下违约率分布柱状图",size=13)
# 设置x,y轴标签
ax1.set_xlabel("maritalStatus")
ax1.set_ylabel("客户人数")
ax2.set_xlabel("maritalStatus")
ax2.set_ylabel("违约率")
#显示汉语标注
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['font.family']=['sans-serif']
#plt.show()

fig,[ax1,ax2] = plt.subplots(1,2,figsize=(16,6))
# 对netLength列的类别设定顺序
data['netLength'] = data['netLength'].astype('category')
data['netLength'] = data['netLength'].cat.set_categories(['0-6个月','6-12个月','12-24个月','24个月以上','无效'],ordered=True)
# 绘制柱状图，查看不同在网时长在不同违约情况的取值分布
sns.countplot(x='netLength', hue='Default1', data=data, ax=ax1)
# 将具体的计数值显示在柱形上方
counts=data['Default1'].groupby(data['netLength']).value_counts().values
count1 = counts[[0,2,4,6,8]]
count2 = counts[[1,3,5,7,9]]
# 将具体的计数值显示在柱形上方
for index, item1, item2 in zip([0,1,2,3,4], count1, count2):
    ax1.text(index-0.2, item1 + 0.05, '%.0f' % item1, ha="center", va= "bottom",fontsize=12)
    ax1.text(index+0.2, item2 + 0.05, '%.0f' % item2, ha="center", va= "bottom",fontsize=12)
# 绘制柱状图查看违约率分布
netLength_rate = data.groupby('netLength')['Default1'].sum() / data.groupby('netLength')['Default1'].count()
sns.barplot(x=[0,1,2,3,4],y=netLength_rate.values,ax=ax2)
# 将具体的计数值显示在柱形上方
for index, item in zip([0,1,2,3,4], netLength_rate):
     ax2.text(index, item, '%.2f' % item, ha="center", va= "bottom",fontsize=12)
# 设置柱形名称
ax1.set_xticklabels(['0-6个月','6-12个月','12-24个月','24个月以上','无效'])
ax2.set_xticklabels(['0-6个月','6-12个月','12-24个月','24个月以上','无效'])
# 设置图例名称
ax1.legend(['未违约','违约'])
# 设置标题以及字体大小
ax1.set_title("不同在网时长下不同违约情况数量分布柱状图",size=13)
ax2.set_title("不同在网时长下违约率分布柱状图",size=13)
# 设置x,y轴标签
ax1.set_xlabel("netLength")
ax1.set_ylabel("客户人数")
ax2.set_xlabel("netLength")
ax2.set_ylabel("违约率")
#显示汉语标注
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['font.family']=['sans-serif']
plt.show()

fig, [ax1,ax2] = plt.subplots(1, 2, figsize=(16, 5))

# 在画布ax1中画出总消费金额的核密度图
sns.kdeplot(data=data['transTotalAmt'],fill=True,ax=ax1)

# 在画布ax2中画出总消费笔数和总消费金额的回归关系图
sns.regplot(x=data['transTotalCnt'],y=data['transTotalAmt'],data=data,ax=ax2)
plt.show()

fig,[ax1,ax2]=plt.subplots(1,2,figsize=(16,6))

#年龄直方图
sns.distplot(a=data['age'],color='red',kde = True,ax=ax1,axlabel='年龄')

#开卡时长直方图

sns.distplot(a=data['card_age'],kde = True,ax=ax2,axlabel = '开卡时长')

ax1.set_title("年龄分布")
ax2.set_title("开卡时长分布")

#显示汉语标注
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['font.family']=['sans-serif']
plt.show()

#总取现笔数和总取现金额的关系

fig,[ax1,ax2] = plt.subplots(1,2,figsize = (16,5))

sns.kdeplot(data = data['cashTotalAmt'],fill = True,ax=ax1,label='总取现金额')

sns.regplot(x=data['cashTotalCnt'],y = data['cashTotalAmt'],data=data,ax=ax2)
plt.show()

#网上消费金额和笔数的关系
fig,[ax1,ax2] = plt.subplots(1,2,figsize = (16,5))

sns.kdeplot(data = data['onlineTransAmt'],fill = True,ax=ax1,label='网上消费金额')

sns.regplot(x=data['onlineTransCnt'],y = data['onlineTransAmt'],data=data,ax=ax2)
ax2.set_xlabel('网上消费笔数')
ax2.set_ylabel('网上消费金额')
plt.show()
