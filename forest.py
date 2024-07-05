import pandas as pd
from sklearn.model_selection import train_test_split
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import sklearn
import cutil
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from joblib import dump


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


#神经网络模型
y = data['Default1']
x = data.drop('Default1',axis=1)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)
x_test.to_csv('C:/Users/zcy/mlp model/test2.csv', index=True)
print("baocunchenggong!@!!!!")
#处理类别不平衡问题，采用smote方法过采样
#smote = SMOTE(sampling_strategy=0.25,random_state=42)
#x_train,y_train = smote.fit_resample(x_train,y_train)
#x_test,y_test = smote.fit_resample(x_test,y_test)

#构建神经网络模型
#调参之前
mlp = MLPClassifier(verbose = True,random_state=333)
mlp.fit(x_train, y_train)
#dump(mlp, 'C:/Users/zcy/mlp model/my_custom_name')
print("保存成功！！！！！")
#打印分数评估
score = mlp.score(x_test, y_test)
print('调参之前的分数：',score)

#调参之后
mlp = MLPClassifier(solver='adam', activation='logistic',verbose = True,hidden_layer_sizes=[10,10,10],
 random_state=333,warm_start = True,early_stopping = True)

start = time.time()
mlp.fit(x_train, y_train)

end = time.time()

print('程序运行时间为: %s Seconds'%(end-start))
#打印分数评估
score = mlp.score(x_test, y_test)
print('调参之后的分数：',score)


#评价神经网络模型
# 使用调参之前
mlp = MLPClassifier(verbose = True,random_state=333)
mlp.fit(x_train, y_train)
y_pred = mlp.predict(x_test)
score = mlp.score(x_test, y_test)
print('得到的分数为:', score)
# 计算roc_auc值，并绘制ROC曲线
# 使用metrics.roc_curve()求出 fpr, tpr, threshold
fpr, tpr, threshold = sklearn.metrics.roc_curve(y_test, y_pred)
# 使用metrics.auc求出roc_auc的值
roc_auc = sklearn.metrics.auc(fpr, tpr)

# 绘制ROC曲线
plt.figure(20)
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.title('Ruc')
plt.show()
# 求出KS值和相应的阈值
ks = max(abs(fpr - tpr))
thre = threshold[abs(fpr - tpr).argmax()]

ks = round(ks * 100, 2)
thre = round(thre, 2)
print('KS值：', ks, '%', '阈值：', thre)

# 绘制真正率TPR与假正率FPR两条线


# 画出KS值的直线
# 将图片大小设为8:6
fig = plt.figure(figsize=(8, 6))
# 将plt.plot里的内容填写完整
plt.plot(threshold[::-1], tpr[::-1], lw=1, alpha=1, label='TPR')
plt.plot(threshold[::-1], fpr[::-1], lw=1, alpha=1, label='FPR')
ks_tpr = tpr[abs(tpr - fpr).argmax()]
ks_fpr = fpr[abs(tpr - fpr).argmax()]
x1 = [thre, thre]
x2 = [ks_fpr, ks_tpr]
plt.plot(x1, x2)

# 设置横纵名称以及图例
plt.xlabel('thresholds')
plt.ylabel('TPR/FPR')
plt.title('KS', fontsize=15)
plt.legend(loc="upper right")
plt.grid(axis='x')

# 在图上标注ks值
plt.annotate('KS', xy=(0.18, 0.45), xytext=(0.25, 0.43),
             fontsize=20, arrowprops=dict(facecolor='green', shrink=0.01))
plt.show()
# 训练集预测概率
y_train_probs = mlp.predict_proba(x_train)[:, 1]
# 测试集预测概率
y_test_probs = mlp.predict_proba(x_test)[:, 1]


def psi(y_train_probs, y_test_probs):
    ## 设定每组的分点
    bins = np.arange(0, 1.1, 0.1)

    ## 将训练集预测概率分组
    y_train_probs_cut = pd.cut(y_train_probs, bins=bins, labels=False)
    ## 计算预期占比
    expect_prop = (pd.Series(y_train_probs_cut).value_counts() / len(y_train_probs)).sort_index()

    ## 将测试集预测概率分组
    y_test_probs_cut = pd.cut(y_test_probs, bins=bins, labels=False)
    ## 计算实际占比
    actual_prop = (pd.Series(y_test_probs_cut).value_counts() / len(y_test_probs)).sort_index()

    ## 计算PSI
    psi = ((actual_prop - expect_prop) * np.log(actual_prop / expect_prop)).sum

    return psi, expect_prop, actual_prop

from sklearn.ensemble import RandomForestClassifier

# 调用psi函数得到psi值
psi, expect_prop, actual_prop = psi(y_train_probs, y_test_probs)
print('psi=', psi)

