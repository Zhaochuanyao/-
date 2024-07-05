
from flask import Flask, request, render_template
import pandas as pd
import joblib  # 用于加载模型
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_and_show_csv():
    if request.method == 'POST':
        csv_file = request.files['csv_file']
        if csv_file:
            data = pd.read_csv(csv_file)
            data = pd.DataFrame(data)
            data = pd.get_dummies(data)
            y = data['Default1'].values
            x = data.drop(['Default1'], axis=1).values
            #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=33, stratify=y)

            model = joblib.load('C:/Users/zcy/mlp model/your_model.joblib')  # 加载模型
            predictions = model.predict(x)
            data['prediction_result'] = predictions
            data = data.head(100)
            return render_template('result.html', data=data)
    return render_template('csv.html')

if __name__ == '__main__':
    app.run()