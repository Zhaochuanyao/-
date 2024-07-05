
from sklearn.model_selection import train_test_split

import seaborn as sns

from sklearn.metrics import roc_auc_score

import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import pandas as pd

from imblearn.over_sampling import SMOTE

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
print(y_predict[:10])
print(len(y_predict))

y_predict_proba = lr.predict_proba(x_test)
print(y_predict_proba[:10])

# 取目标分数为正类(1)的概率估计
y_predict = y_predict_proba[:,1]

# 利用roc_auc_score查看模型效果
test_auc = roc_auc_score(y_test,y_predict)


print('逻辑回归模型 test_auc:',test_auc)
#0.6571021579514986

#模型优化


# 建立一个LogisticRegression对象，命名为lr
lr = LogisticRegression(C=0.6, class_weight='balanced', penalty='l2')

# 对lr对象调用fit方法，带入训练集x_train, y_train进行训练
lr.fit(x_train, y_train)

# 对训练好的lr模型调用predict_proba方法
y_predict = lr.predict_proba(x_test)[:, 1]

# 调用roc_auc_score方法
test_auc = roc_auc_score(y_test, y_predict)

print('逻辑回归模型test auc:')
print(test_auc)
#0.7596808148328698

#使用标准化模型提升效果
continuous_columns = ['age','cashTotalAmt','cashTotalCnt','monthCardLargeAmt','onlineTransAmt','onlineTransCnt','publicPayAmt','publicPayCnt','transTotalAmt','transTotalCnt','transCnt_non_null_months','transAmt_mean','transAmt_non_null_months','cashCnt_mean','cashCnt_non_null_months','cashAmt_mean','cashAmt_non_null_months','card_age', 'trans_total','total_withdraw', 'avg_per_withdraw','avg_per_online_spend', 'avg_per_public_spend', 'bad_record','transCnt_mean','noTransWeekPre']

# 对data中所有连续型的列进行Z-score标准化

data[continuous_columns]=data[continuous_columns].apply(lambda x:(x-x.mean())/x.std())

# 查看标准化后的数据的均值和标准差，以cashAmt_mean为例
print('cashAmt_mean标准化后的均值：',data['cashAmt_mean'].mean())
print('cashAmt_mean标准化后的标准差：',data['cashAmt_mean'].std())

# 查看标准化后对模型的效果提升
y = data['Default1'].values
x = data.drop(['Default1'], axis=1).values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state = 33,stratify=y)


lr = LogisticRegression(penalty='l2',C=0.6,class_weight='balanced')
lr.fit(x_train, y_train)

# 查看模型预测结果
y_predict = lr.predict_proba(x_test)[:,1]
auc_score =roc_auc_score(y_test,y_predict)
print('score:',auc_score)
#离散化数据提升效果
continuous_columns = ['age','cashTotalAmt','cashTotalCnt','monthCardLargeAmt','onlineTransAmt','onlineTransCnt','publicPayAmt','publicPayCnt','transTotalAmt','transTotalCnt','transCnt_non_null_months','transAmt_mean','transAmt_non_null_months','cashCnt_mean','cashCnt_non_null_months','cashAmt_mean','cashAmt_non_null_months','card_age', 'trans_total','total_withdraw', 'avg_per_withdraw','avg_per_online_spend', 'avg_per_public_spend', 'bad_record','transCnt_mean','noTransWeekPre']

# 对data中数值连续型的列进行等频离散化，将每一列都离散为5个组。
data[continuous_columns] = data[continuous_columns].apply(lambda x : pd.qcut(x,5,duplicates='drop'))


# 查看离散化后的数据
print(data.head())
data.to_csv('C:/Users/zcy/mlp model/test3.csv', index=False)
# 查看离散化后对模型的效果提升
# 先对各离散组进行One-Hot处理
data=pd.get_dummies(data)
y = data['Default1'].values
x = data.drop(['Default1'], axis=1).values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state = 33,stratify=y)



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

lr = LogisticRegression(penalty='l1', solver='saga',C=0.6,class_weight='balanced')
start = time.time()
lr.fit(x_train, y_train)
end = time.time()

print('程序运行时间为: %s Seconds'%(end-start))
# 查看模型预测结果
y_predict = lr.predict_proba(x_test)[:,1]
score_auc = roc_auc_score(y_test,y_predict)
print('score:',score_auc)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

rf_clf = RandomForestClassifier()
rf_clf.fit(x_train,y_train)
y_predict = rf_clf.predict_proba(x_test)[:,1]


# 查看模型效果
test_auc = roc_auc_score(y_test,y_predict)
print ("AUC Score (Test): %f" % test_auc)

# 尝试设置参数n_estimators
rf_clf1 =  RandomForestClassifier(n_estimators=100)

rf_clf1.fit(x_train, y_train)
y_predict1 = rf_clf1.predict_proba(x_test)[:,1]

# 查看模型效果
test_auc = roc_auc_score(y_test,y_predict1)
print ("AUC Score (Test): %f" % test_auc)


#参数调优
# 定义存储AUC分数的数组
scores_train=[]
scores_test=[]
# 定义存储n_estimators取值的数组
estimators=[]

# 设置n_estimators在100-210中每隔20取一个数值
for i in range(100,210,20):
        estimators.append(i)
        rf = RandomForestClassifier(n_estimators=i, random_state=12)
        rf.fit(x_train,y_train)

        y_predict = rf.predict_proba(x_test)[:,1]
        scores_test.append(roc_auc_score(y_test,y_predict))

# 查看我们使用的n_estimators取值
print("estimators =", estimators)

# 查看以上模型中在测试集最好的评分
print("best_scores_test =",max(scores_test))

# 画出n_estimators与AUC的图形
fig,ax = plt.subplots()

# 设置x y坐标名称
ax.set_xlabel('estimators')
ax.set_ylabel('AUC分数')
plt.plot(estimators,scores_test, label='测试集')

#显示汉语标注
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['font.family']=['sans-serif']

# 设置图例
plt.legend(loc="lower right")
plt.show()

#



from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
rf = RandomForestClassifier()

# 设置需要调试的参数
tuned_parameters = {
    'n_estimators': [180,200],
    'criterion': ['entropy', 'gini'],
    'max_depth': [8, 10],
    'min_samples_split': [2, 3]
}

# 调用网格搜索函数
rf_clf = GridSearchCV(rf, tuned_parameters, cv=2, n_jobs=2, scoring='roc_auc')
start = time.time()
rf_clf.fit(x_train, y_train)
dump(rf_clf, 'C:/Users/zcy/mlp model/your_model.joblib')
end = time.time()



print('程序运行时间为: %s Seconds'%(end-start))
y_predict = rf_clf.predict_proba(x_test)[:, 1]
test_auc = roc_auc_score(y_test, y_predict)
print('随机森林模型test AUC:')
print(test_auc)

#模型评估

import sklearn

# 用metrics.roc_curve()求出 fpr, tpr, threshold

# y_predict_best=???用到逻辑回归参数调优
y_predict_best = y_predict

fpr, tpr, threshold = sklearn.metrics.roc_curve(y_test, y_predict_best)

# 用metrics.auc求出roc_auc的值
roc_auc = sklearn.metrics.auc(fpr, tpr)

# 将图片大小设为8:6
fig, ax = plt.subplots(figsize=(8, 6))

# 将plt.plot里的内容填写完整
plt.plot(fpr, tpr, label='AUC = %0.2f' % roc_auc)

# 将图例显示在右下方
plt.legend(loc='lower right')

# 画出一条红色对角虚线
plt.plot([0, 1], [0, 1], 'r--')

# 设置横纵坐标轴范围
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])

# 设置横纵名称以及图形名称
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.show()

#森林
# 用metrics.roc_curve()求出 fpr, tpr, threshold
fpr, tpr, threshold = sklearn.metrics.roc_curve(y_test, y_predict_best)

# 用metrics.auc求出roc_auc的值
roc_auc = sklearn.metrics.auc(fpr, tpr)

# 将图片大小设为8:6
fig, ax = plt.subplots(figsize=(8, 6))

# 将plt.plot里的内容填写完整
plt.plot(fpr, tpr, label='AUC = %0.2f' % roc_auc)

# 将图例显示在右下方
plt.legend(loc='lower right')

# 画出一条红色对角虚线
plt.plot([0, 1], [0, 1], 'r--')

# 设置横纵坐标轴范围
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])

# 设置横纵名称以及图形名称
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.show()

#使用ks值进行评估
# 用metric.roc_curve()求出 fpr, tpr, threshold
fpr, tpr, threshold = sklearn.metrics.roc_curve(y_test, y_predict_best)
# print(threshold)
# 求出KS值和相应的阈值
ks = max(abs(tpr - fpr))
thre = threshold[abs(tpr - fpr).argmax()]

ks = round(ks * 100, 2)
thre = round(thre, 2)
print('KS值：', ks, '%', '阈值：', thre)

# 将图片大小设为8:6
fig = plt.figure(figsize=(8, 6))
# 将plt.plot里的内容填写完整

plt.plot(threshold[::-1], tpr[::-1], lw=1, alpha=1, label='真正率TPR')
plt.plot(threshold[::-1], fpr[::-1], lw=1, alpha=1, label='假正率FPR')

# 画出KS值的直线
ks_tpr = tpr[abs(tpr - fpr).argmax()]
ks_fpr = fpr[abs(tpr - fpr).argmax()]
x1 = [thre, thre]
x2 = [ks_fpr, ks_tpr]
plt.plot(x1, x2)

# 设置横纵名称以及图例
plt.xlabel('阈值')
plt.ylabel('真正率TPR/假正率FPR')
plt.title('KS曲线', fontsize=15)
plt.legend(loc="upper right")
plt.grid(axis='x')

# 在图上标注ks值
plt.annotate('KS值', xy=(0.18, 0.45), xytext=(0.25, 0.43),
             fontsize=20, arrowprops=dict(facecolor='green', shrink=0.01))


#森林
#用metric.roc_curve()求出 fpr, tpr, threshold
fpr, tpr, threshold = sklearn.metrics.roc_curve(y_test, y_predict_best)
#处理类别不平衡问题，采用smote方法过采样
smote = SMOTE(sampling_strategy=0.25,random_state=42)
x_train,y_train = smote.fit_resample(x_train,y_train)
x_test,y_test = smote.fit_resample(x_test,y_test)
#求出KS值和相应的阈值
ks = max(abs(tpr-fpr))
thre = threshold[abs(tpr-fpr).argmax()]


ks = round(ks*100, 2)
thre = round(thre, 2)
print('KS值：', ks,  '%', '阈值：', thre)

#将图片大小设为8:6
fig = plt.figure(figsize=(8,6))
#将plt.plot里的内容填写完整

plt.plot(threshold[::-1], tpr[::-1], lw=1, alpha=1,label='真正率TPR')
plt.plot(threshold[::-1], fpr[::-1], lw=1, alpha=1,label='假正率FPR')


#画出KS值的直线
ks_tpr = tpr[abs(tpr-fpr).argmax()]
ks_fpr = fpr[abs(tpr-fpr).argmax()]
x1 = [thre, thre]
x2 = [ks_fpr, ks_tpr]
plt.plot(x1, x2)

#设置横纵名称以及图例
plt.xlabel('阈值')
plt.ylabel('真正率TPR/假正率FPR')
plt.title('KS曲线', fontsize=15)
plt.legend(loc="upper right")
plt.grid(axis='x')

# 在图上标注ks值
plt.annotate('KS值', xy=(0.26, 0.45), xytext=(0.30, 0.43),
             fontsize=20,arrowprops=dict(facecolor='green', shrink=0.01))


#计算权重（逻辑回归模型）
from sklearn.linear_model import LogisticRegression

lr_clf = LogisticRegression(penalty='l2', C=0.6, random_state=55)
lr_clf.fit(x_train, y_train)

# 查看逻辑回归各项指标系数
coefficient = lr_clf.coef_

# 取出指标系数，并对其求绝对值
importance = abs(coefficient)

# 通过图形的方式直观展现前八名的重要指标
index = data.drop('Default1', axis=1).columns
feature_importance = pd.DataFrame(importance.T, index=index).sort_values(by=0, ascending=True)

# # 查看指标重要度
print(feature_importance)

# 水平条形图绘制
feature_importance.tail(8).plot(kind='barh', title='Feature Importances', figsize=(8, 6), legend=False)
plt.show()

#森林

rf = RandomForestClassifier(n_estimators = 150, criterion = 'entropy', max_depth = 5, min_samples_split = 2, random_state=12)
rf.fit(x_train, y_train)

# 查看随机森林各项指标系数
importance = rf.feature_importances_

# 通过图形的方式直观展现前八名的重要指标
index=data.drop('Default1', axis=1).columns
feature_importance = pd.DataFrame(importance.T, index=index).sort_values(by=0, ascending=True)

# # 查看指标重要度
print(feature_importance)

# 水平条形图绘制
feature_importance.tail(8).plot(kind='barh', title='Feature Importances', figsize=(8, 6), legend=False)
plt.show()



