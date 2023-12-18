from flask import Flask, request
import json
import os
import tensorflow as tf
import numpy as np

app = Flask(__name__)

@app.route('/interest')
def interest():
    money = float(request.args.get('money', 0))

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
    predicted_rates = predicted_rates.tolist()

    # money * predicted_rates
    calculated_money = list(map(lambda x: x * money, predicted_rates))

    print(predicted_rates)
    print(type(predicted_rates))

    return {
        "data": {
            "rates": predicted_rates,
            "calculated": calculated_money
        }
    }

@app.route('/gold')
def gold():
    max_price = 1985.0
    min_price = 35.21

    # Load the Model
    model = tf.keras.models.load_model("./gold_model")

    future_steps = 12
    array_to_reshape = [1.27777481, 1.31062317, 1.34283304]
    new_X = np.array(array_to_reshape).reshape(1, 3, 1)
    predicted_normalized = []
    for _ in range(future_steps):
        pred = model.predict(new_X)
        predicted_normalized.append(pred[0, 0])
        new_X = np.roll(new_X, -1, axis=1)
        new_X[0, -1, 0] = pred[0, 0]

    # Reverse the normalization
    predicted_prices = np.array(predicted_normalized) * (max_price - min_price) + min_price

    print(predicted_prices)
    return {"data": predicted_prices.tolist()}

@app.route('/house')
def house():
    return {"data": [
        5.697893142700195,
        5.694507598876953,
        5.691179275512695,
        5.687906265258789,
        5.684688568115234,
        5.681525707244873,
        5.6784162521362305,
        5.675358295440674,
        5.672351837158203,
        5.66939640045166,
        5.6664910316467285,
        5.663634300231934
    ]}

@app.route('/stock')
def stock():
    return {"data": [
        5.697893142700195,
        5.694507598876953,
        5.691179275512695,
        5.687906265258789,
        5.684688568115234,
        5.681525707244873,
        5.6784162521362305,
        5.675358295440674,
        5.672351837158203,
        5.66939640045166,
        5.6664910316467285,
        5.663634300231934
    ]}

if __name__ == '__main__':
    app.run(debug=True)