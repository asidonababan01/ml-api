from flask import Flask
import json
import os
import tensorflow as tf
import numpy as np

app = Flask(__name__)

@app.route('/interest')
def interest():
    max_rate = 6.0
    min_rate = 3.5

    # Load the Model
    model = tf.keras.models.load_model(os.environ.get("INTEREST_PATH"))

    future_steps = 12
    array_to_reshape = [0.88335973, 0.88193572, 0.88053465]
    new_X = np.array(array_to_reshape).reshape(1, 3, 1)
    predicted_normalized = []
    for _ in range(future_steps):
        pred = model.predict(new_X)
        predicted_normalized.append(pred[0, 0])
        new_X = np.roll(new_X, -1, axis=1)
        new_X[0, -1, 0] = pred[0, 0]

    # Reverse the normalization
    predicted_rates = np.array(predicted_normalized) * (max_rate - min_rate) + min_rate

    # money * predicted_rates

    print(predicted_rates)

    return {"data": json.dumps(predicted_rates.tolist())}

if __name__ == '__main__':
    app.run(debug=True)