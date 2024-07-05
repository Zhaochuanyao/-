import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import sklearn
import time
from imblearn.over_sampling import SMOTE

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



#数据预处理

na_counts=data.isnull().sum()

missing_value = na_counts[na_counts>0].sort_values(axis = 0,ascending= False)

print(missing_value)

filling_columns = ['maritalStatus','threeVerify','education']

for column in filling_columns:
    data[column] = data[column].astype('category')
    data[column] = data[column].cat.add_categories(['未知'])
for column in filling_columns:
    data[column].fillna('未知',inplace=True)
#删除sex的缺失值
data = data.dropna(subset=['sex'])

na_counts=data.isnull().sum()

missing_value = na_counts[na_counts>0].sort_values(axis = 0,ascending= False)

print(missing_value)
#解决离散型异常值
print(data[data['isCrime']>1]['isCrime'])
data['isCrime'] = data['isCrime'].replace(2,0)
print(data['isCrime'].value_counts())

# 所有连续型特征列名已保存在continuous_columns中
continuous_columns = ['age','cashTotalAmt','cashTotalCnt','monthCardLargeAmt','onlineTransAmt','onlineTransCnt','publicPayAmt','publicPayCnt','transTotalAmt','transTotalCnt','transCnt_non_null_months','transAmt_mean','transAmt_non_null_months','cashCnt_mean','cashCnt_non_null_months','cashAmt_mean','cashAmt_non_null_months','card_age']
# 查看数据各连续型特征的最小值
data_con_min = data[continuous_columns].min()
print(data_con_min)

online_trans = data[data['onlineTransAmt'] < 0][['onlineTransAmt','onlineTransCnt']]
print(online_trans)

data.loc[data['onlineTransCnt'] == 0,'onlineTransAmt'] = 0
# 查看修正后网上消费笔数为0时，网上消费金额与网上消费笔数
online_after = data[data["onlineTransCnt"]  == 0 ][["onlineTransAmt","onlineTransCnt"]]
print(online_after)

data = data[data['onlineTransAmt']<2.0e+07]
print(data.head())
# 从原始数据中筛选出公共事业缴费金额小于0时，公共事业缴费笔数和公共事业缴费金额这两列
public_pay = data[data['publicPayAmt']<0][["publicPayCnt","publicPayAmt"]]
print(public_pay)
data.loc[data['publicPayCnt']==0,'publicPayAmt']=0

# 查看修正后的，公共事业缴费笔数为0时的公共事业缴费金额与公共事业缴费笔数
public_after = data[data["publicPayCnt"]  ==  0][["publicPayAmt","publicPayCnt"]]
print(public_after)

fig,ax  =plt.subplots(figsize = (16,5))
sns.boxplot(data['publicPayAmt'],ax=ax,orient='v')
plt.title('公共事业缴费金额数据分布')
plt.show()

public_pay = data[data['publicPayAmt'] < -4.0e+06]
print(public_pay[['publicPayCnt','publicPayAmt']])
# 从原始数据中筛选出总消费笔数等于0时，总消费笔数，总消费金额这两列
transTotal = data[data["transTotalCnt"]  ==  0][["transTotalCnt","transTotalAmt"]]
print(transTotal)
#并无异常值

fig,ax = plt.subplots(figsize=(8,6))
# 绘制盒图，查看总消费金额数据分布。
sns.boxplot(data['transTotalAmt'],ax=ax,orient='v')
plt.title('transTotalAmt distribution')
plt.show()

transTotal = data[data['transTotalAmt']>1.0e+07]
print(transTotal[['transTotalAmt','transTotalCnt','onlineTransAmt','onlineTransCnt','monthCardLargeAmt']])

fig,ax = plt.subplots(figsize=(8,6))
# 绘制盒图，查看总取现金额数据分布。
sns.boxplot(data['cashTotalAmt'],ax=ax,orient='v')
plt.title('cashTotalAmt distribution')
plt.show()

cashTotal = data[data['cashTotalAmt'] > 5.0e+05]
print(cashTotal)

fig,ax = plt.subplots(figsize=(8,6))
# 绘制盒图，查看月最大消费金额数据分布
sns.boxplot(data['monthCardLargeAmt'],ax=ax,orient='v')
plt.title('monthCardLargeAmt distribution')
plt.show()

monthCard = data[data['monthCardLargeAmt'] > 2.0e+06]
print(monthCard)

fig,ax = plt.subplots(figsize=(8,6))
# 绘制盒图，查看总消费笔数数据分布
sns.boxplot(data['transTotalCnt'],ax=ax,orient='v')
plt.title('transTotalCnt distribution')
plt.show()

data = data[data['transTotalCnt'] < 6000]
print(data.head())


data["maritalStatus"] = data["maritalStatus"].map({"未知":0,"未婚":1,"已婚":2})
data['education']= data['education'].map({"未知":0,"小学":1,"初中":2,"高中":3,"本科以上":4})
data['idVerify']= data['idVerify'].map({"未知":0,"一致":1,"不一致":2})
data['threeVerify']= data['threeVerify'].map({"未知":0,"一致":1,"不一致":2})
data["netLength"] = data['netLength'].map({"无效":0,"0-6个月":1,"6-12个月":2,"12-24个月":3,"24个月以上":4})
data["sex"] = data['sex'].map({"未知":0,"男":1,"女":2})
data["CityId"] = data['CityId'].map({"一线城市":1,"二线城市":2,"其它":3})

print(data.head())

#独热编码

data=pd.get_dummies(data = data,columns = ['maritalStatus','education','idVerify','threeVerify','Han','netLength','sex','CityId'])

print(data.columns)

#信用评估指标体系构建

# 计算客户年消费总额。
trans_total =data['transCnt_mean']*data['transAmt_mean']


# 将计算结果保留到小数点后六位。
trans_total =round(trans_total,6)


# 将结果加在data数据集中的最后一列，并将此列命名为trans_total。
data['trans_total'] =trans_total


print(data['trans_total'].head(20))

# 计算客户年取现总额。
total_withdraw =data['cashCnt_mean']*data['cashAmt_mean']

# 将计算结果保留到小数点后六位。
total_withdraw =round(total_withdraw,6)

# 将结果加在data数据集的最后一列，并将此列命名为total_withdraw。
data['total_withdraw'] =total_withdraw

print(data['total_withdraw'].head(20))


# 计算客户的平均每笔取现金额。
avg_per_withdraw = data['cashTotalAmt']/data['cashTotalCnt']

avg_per_withdraw = avg_per_withdraw.replace([np.inf,np.nan],0)

avg_per_withdraw = round(avg_per_withdraw,6)

data['avg_per_withdraw']=avg_per_withdraw

print(data['avg_per_withdraw'].head(20))


# 请计算客户的网上平均每笔消费额。
avg_per_online_spend = data['onlineTransAmt'] / data['onlineTransCnt']

# 将所有的inf和NaN变为0。
avg_per_online_spend = avg_per_online_spend.replace([np.inf,np.nan],0)

# 将计算结果保留到小数点后六位。
avg_per_online_spend =round(avg_per_online_spend,6)

# 将结果加在data数据集的最后一列，并将此列命名为avg_per_online_spend。
data['avg_per_online_spend'] =avg_per_online_spend

print(data['avg_per_online_spend'].head(20))

# 请计算客户的公共事业平均每笔缴费额。
avg_per_public_spend =data['publicPayAmt'] / data['publicPayCnt']

# 将所有的inf和NaN变为0。
avg_per_public_spend = avg_per_public_spend.replace([np.inf,np.nan],0)

# 将计算结果保留到小数点后六位。
avg_per_public_spend =round(avg_per_public_spend,6)

# 将结果加在data数据集的最后一列，并将此列命名为avg_per_public_spend。
data['avg_per_public_spend'] =avg_per_public_spend

print(data['avg_per_public_spend'].head(20))

#请计算客户的不良记录分数。
bad_record =data['inCourt'] + data['isDue'] + data['isCrime'] + data['isBlackList']

#将计算结果加在data数据集的最后一列，并将此列命名为bad_record。
data['bad_record'] =bad_record

print(data['bad_record'].head(20))

#构建风控模型
# 筛选data中的Default列的值，赋予变量y
y = data['Default1'].values

# 筛选除去Default列的其他列的值，赋予变量x
x = data.drop(['Default1'], axis=1).values

# 使用train_test_split方法，将x,y划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=33,stratify=y)

# 查看划分后的x_train与x_test的长度
len_x_train = len(x_train)
len_x_test = len(x_test)
print('x_train length: %d, x_test length: %d'%(len_x_train,len_x_test))

# 查看分层采样后的训练集中违约客户人数的占比
train_ratio = y_train.sum()/len(y_train)
print(train_ratio)

# 查看分层采样后的测试集中违约客户人数的占比
test_ratio = y_test.sum()/len(y_test)
print(test_ratio)

# 调用模型，新建模型对象
lr = LogisticRegression()

# 带入训练集x_train, y_train进行训练
lr.fit(x_train, y_train)

# 对训练好的lr模型调用predict方法,带入测试集x_test进行预测
y_predict = lr.predict(x_test)

# 查看模型预测结果
print(y_predict[:100])
print(len(y_predict))

y_predict_proba = lr.predict_proba(x_test)
print(y_predict_proba[:10])