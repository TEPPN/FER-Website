from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import base64
from tensorflow import keras

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from HTML page

# Load your model and face detector
model = keras.models.load_model('model_file_100epochs.h5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Emotion label dictionary
label_dict = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Neutral',
    5: 'Sad',
    6: 'Surprise'
}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'No image data'}), 400

    image_data = data['image'].split(',')[1]  # remove "data:image/jpeg;base64,"
    img_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 3)

    if len(faces) == 0:
        return jsonify({'emotion': 'No face detected'})

    # Take the first detected face
    x, y, w, h = faces[0]
    sub_face = gray[y:y+h, x:x+w]
    resized = cv2.resize(sub_face, (48, 48))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, 48, 48, 1))

    result = model.predict(reshaped)
    label = np.argmax(result, axis=1)[0]
    emotion = label_dict[label]

    return jsonify({'emotion': emotion})

if __name__ == '__main__':
    app.run(debug=True)
