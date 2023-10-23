from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from keras.preprocessing import image
from keras.models import load_model
from werkzeug.utils import secure_filename
import os
import numpy as np
import tensorflow as tf

app = Flask(__name__)
# Yüklenen dosyaların saklandığı klasörün adı
app.config['UPLOAD_FOLDER'] = 'Test'
model = load_model('modelim.h5', compile=True)

# TensorFlow Lite modelini yükleyin
tflite_model = tf.lite.Interpreter(model_path='model.tflite')
tflite_model.allocate_tensors()

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def classify_image():
    predicted_class = None
    uploaded_filename = None

    if request.method == 'POST':
        # Gelen dosyayı işle
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            # Dosyayı güvenli bir şekilde kaydet
            filename = secure_filename(uploaded_file.filename)
            uploaded_file.save(os.path.join(
                app.config['UPLOAD_FOLDER'], filename))

            # Kaydedilen dosyayı işle
            img = image.load_img(os.path.join(
                app.config['UPLOAD_FOLDER'], filename), target_size=(64, 64))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)

            # TensorFlow Lite modelini kullanarak tahmin yapın
            tflite_model.set_tensor(tflite_model.get_input_details()[0]['index'], img)
            tflite_model.invoke()
            predictions = tflite_model.get_tensor(tflite_model.get_output_details()[0]['index'])
            predicted_class = int(predictions[0][0])
            uploaded_filename = filename

    return render_template('index.html', predicted_class=predicted_class, uploaded_filename=uploaded_filename)

if __name__ == '__main__':
    app.run(debug=False)
