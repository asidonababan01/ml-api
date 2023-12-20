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
    min_rate = 0.95636
    max_rate = 13.5061

    # Load the Model
    model = tf.keras.models.load_model("./housepricegrowth_rate_model")

    # Forecast future growth rates
    future_steps = 12
    array_to_reshape = [0.1603778, 0.16483852, 0.16887896]
    new_X = np.array(array_to_reshape).reshape(1, 3, 1)
    predicted_normalized = []
    for _ in range(future_steps):
        pred = model.predict(new_X)
        predicted_normalized.append(pred[0, 0])
        new_X = np.roll(new_X, -1, axis=1)
        new_X[0, -1, 0] = pred[0, 0]

    # Reverse the normalization
    predicted_rates = np.array(predicted_normalized) * (max_rate - min_rate) + min_rate

    # percent kenaikan

    print(predicted_rates)
    return {"data": predicted_rates.tolist()}

@app.route('/stock')
def stock():
    min_price = 732.401001
    max_price = 7228.914063

    # Load the Model
    model = tf.keras.models.load_model("./stock_model")

    # Forecast future stock prices
    future_steps = 12
    array_to_reshape = [
        0.94176954, 0.94003364, 0.94063349, 0.93479006, 0.95178979,
        0.90831192, 0.912717, 0.95419771, 0.95756888, 0.95551121,
        0.92662169, 0.97719194, 0.94432652, 0.94072467, 0.93620187,
        0.93090552, 0.92500818, 0.91865826, 0.91202885, 0.90526968,
        0.89843303, 0.89155924, 0.88477463, 0.87809634
    ]
    new_X = np.array(array_to_reshape).reshape(1, 24, 1)
    predicted_normalized = []
    for _ in range(future_steps):
        pred = model.predict(new_X)
        predicted_normalized.append(pred[0, 0])
        new_X = np.roll(new_X, -1, axis=1)
        new_X[0, -1, 0] = pred[0, 0]

    # Reverse the normalization
    predicted_prices = np.array(predicted_normalized) * (max_price - min_price) + min_price
    
    return {"data": predicted_prices.tolist()}

if __name__ == '__main__':
    app.run(debug=True)