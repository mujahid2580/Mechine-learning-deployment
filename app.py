from flask import Flask, request, url_for, redirect, render_template
import joblib
import numpy as np

# We need create an application

app = Flask(__name__,template_folder='templates')
model = joblib.load('Random_forest_marriage_prediction_model.ml')

#shameer = [['male',5.7,'Muslim','Shaik','Urdu','Software','Madanapalle','India']]
#New_data = np.array([[2,1,2,34,6,22,153,19,170.18]])
#print(model.predict(New_data))
@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    print(int_features)
    final = np.array([[int_features]])
    predication = model.predict(final)
    print(predication)
    output='{0:.{1}f}'.format(predication[0][1], 2)
    return render_template('index.html',pred='In this Age You will Get a Marriage.\nProbability of age occuring is {}'.format(output))
   

if __name__ == '__main__':
    app.run()