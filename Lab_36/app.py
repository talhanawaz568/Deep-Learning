from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load the model once when the server starts
model = tf.keras.models.load_model('mnist_deployment_model.keras')

@app.route('/')
def home():
    return "MNIST Inference API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request
        data = request.get_json(force=True)
        # Convert list to numpy array and ensure correct shape (1, 28, 28)
        input_data = np.array(data['input']).reshape(1, 28, 28)
        
        # Perform inference
        prediction = model.predict(input_data, verbose=0)
        predicted_class = int(np.argmax(prediction))
        
        return jsonify({
            'status': 'success',
            'prediction': predicted_class,
            'confidence': float(np.max(prediction))
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    # Running on port 5000
    app.run(debug=True)
