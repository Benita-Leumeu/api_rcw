from flask import Flask, request, json
import pickle
import pandas as pd

# Import ML model
model = pickle.load(open('final_model.pickle', 'rb'))
# App flask() == App express()
app = Flask(__name__)

# Define middleware
@app.route('/prediction', methods = ['POST'])
def reply_to_client():
    # Get client request
    data = request.get_json(force=True)
    # preg = data['preg']
    # print('Preg:', preg)
    # print(f'Client sent: {data}')
    dataframe = pd.DataFrame(data, index=[0])
    # print(dataframe)
    # Get prediction from the model and convert into int
    prediction = int(model.predict(dataframe))
    '''
    probability = model.predict_proba(dataframe)
    probability = probability[0]
    probability = probability[prediction]
    '''
    probability = (model.predict_proba(dataframe)[0][prediction]).round(3)
    print(f'Class: {prediction} - Proba: {probability}')
    output = json.dumps({'class': prediction, 'proba': probability})
    return output

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8000, debug=True)
    
