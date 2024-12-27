from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
app = Flask(__name__)

# Load your decision tree model
with open(r'decision_tree_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    try:
        data = request.json
        weight = data['weights']
         
        new_data = pd.DataFrame({
        'Symptom_1': [weight[0]],
        'Symptom_2': [weight[1]],
        'Symptom_3': [weight[2]],
        'Symptom_4': [weight[3]],
        'Symptom_5': [weight[4]],
        'Symptom_6': [weight[5]],
        'Symptom_7': [weight[6]],
        'Symptom_8': [weight[7]],
        'Symptom_9': [weight[8]],
        'Symptom_10': [weight[9]],
        'Symptom_11': [weight[10]],
        'Symptom_12': [weight[11]],
        'Symptom_13': [weight[12]],
        'Symptom_14': [weight[13]],
        'Symptom_15': [weight[14]],
        'Symptom_16': [weight[15]],
        'Symptom_17': [weight[16]]
         })
        result = loaded_model.predict(new_data)
        result_list = result.tolist()

        # Print the list
        print("Predicted Labels:", result_list)

        # Return the list as JSON response
        return jsonify(result_list)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run()
