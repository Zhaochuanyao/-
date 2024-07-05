import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
# 加载训练好的模型
model = load_model('C:/Users/zcy/mlp model/my_custom_name')

# 准备输入数据（根据模型的输入要求进行准备）
data = pd.read_csv('C:/Users/zcy/mlp model/test.csv')

# 进行预测
predictions = model.predict(data)

# 打印预测结果
print(predictions)