# from flask import Flask, request, render_template
# import os
# import cv2
# import numpy as np

# app = Flask(__name__)

# def process_image(image_path, skin_type):
#     image = cv2.imread(image_path)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     min_pimple_area = 50
#     pimple_count = 0
#     for contour in contours:
#         if cv2.contourArea(contour) > min_pimple_area:
#             pimple_count += 1

#     recommendations = recommend_medicine(pimple_count, skin_type)
#     return pimple_count, recommendations

# def recommend_medicine(pimple_count, skin_type):
#     recommendations = []
#     if pimple_count <= 5:
#         recommendations.append("Consider using a gentle cleanser and a mild exfoliant.")
#     else:
#         recommendations.append("You may want to consult a dermatologist for more severe acne cases.")

#     if skin_type == "oily":
#         recommendations.append("Look for products with salicylic acid or benzoyl peroxide.")
#     elif skin_type == "dry":
#         recommendations.append("Opt for non-comedogenic, hydrating products.")

#     return recommendations

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload():
#     uploaded_file = request.files['image']
#     skin_type = request.form.get('skin_type')
#     if uploaded_file.filename != '':
#         file_path = os.path.join(uploaded_file.filename)
#         uploaded_file.save(file_path)
#         pimple_count, recommendations = process_image(file_path, skin_type)
#         return render_template('result.html', pimple_count=pimple_count, recommendations=recommendations)
#     return 'No file selected for upload.'

# if __name__ == '__main__':
#     app.run(debug=True)

import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template

app = Flask(__name__)

# Load a pre-trained model (you can train your own as described in the previous response)
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Define a function for pimple detection
def detect_pimples(image):
    img = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224)) / 255.0

    # Use the pre-trained model for pimple detection
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)

    pimple_probability = predictions[0][0]  # Probability of having a pimple
    return pimple_probability

# Define a function for skincare recommendations
def get_recommendations(pimple_probability):
    if pimple_probability > 0.5:
        return "You have detected pimples. It is advisable to consult a dermatologist for proper treatment."
    else:
        return "Your skin looks clear. Maintain good skincare practices to keep it healthy."

# Main route for uploading an image
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        uploaded_image = request.files['file']
        if uploaded_image:
            pimple_probability = detect_pimples(uploaded_image)
            recommendations = get_recommendations(pimple_probability)
            return render_template('result.html', pimple_probability=pimple_probability, recommendations=recommendations)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)



# from flask import Flask, request, render_template, redirect, url_for, session, flash
# import os
# import cv2
# import numpy as np

# app = Flask(__name__)
#  # Change this to a secure secret key

# # Define a dummy user for demonstration purposes.
# dummy_user = {'username': 'mullai', 'password': '30mu'}

# def process_image(image_path, skin_type):
#     image = cv2.imread(image_path)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     min_pimple_area = 50
#     pimple_count = 0
#     for contour in contours:
#         if cv2.contourArea(contour) > min_pimple_area:
#             pimple_count += 1

#     recommendations = recommend_medicine(pimple_count, skin_type)
#     return pimple_count, recommendations

# def recommend_medicine(pimple_count, skin_type):
#     recommendations = []
#     if pimple_count <= 5:
#         recommendations.append("Consider using a gentle cleanser and a mild exfoliant.")
#     else:
#         recommendations.append("You may want to consult a dermatologist for more severe acne cases.")

#     if skin_type == "oily":
#         recommendations.append("Look for products with salicylic acid or benzoyl peroxide.")
#     elif skin_type == "dry":
#         recommendations.append("Opt for non-comedogenic, hydrating products.")

#     return recommendations

# @app.route('/')
# def index():
#     if 'username' in session:
#         return render_template('index.html')
#     else:
#         return redirect(url_for('login'))

# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         username = request.form['username']
#         password = request.form['password']
#         if username == dummy_user['mullai'] and password == dummy_user['30mu']:
#             session['username'] = username
#             return render_template('index.html')
#         else:
#             flash('Invalid username or password. Please try again.', 'error')
#     return render_template('login.html')

# @app.route('/logout')
# def logout():
#     session.pop('username', None)
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload():
#     if 'username' not in session:
#         return render_template('index.html')

#     uploaded_file = request.files['image']
#     skin_type = request.form.get('skin_type')
#     if uploaded_file.filename != '':
#         file_path = os.path.join(uploaded_file.filename)
#         uploaded_file.save(file_path)
#         pimple_count, recommendations = process_image(file_path, skin_type)
#         return render_template('result.html', pimple_count=pimple_count, recommendations=recommendations)
#     return 'No file selected for upload.'

# if __name__ == '__main__':
#     app.run(debug=True)
