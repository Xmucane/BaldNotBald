from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import os
import numpy as np
from PIL import Image

app = Flask(__name__)
app.template_folder = 'templates'
# Yüklenen dosyaların saklandığı klasörün adı
app.config['UPLOAD_FOLDER'] = 'Test'
model = load_model('modelim.h5')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def classify_image():
    if request.method == 'POST':
        # Gelen dosyayı işle
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            # Dosyayı güvenli bir şekilde kaydet
            filename = secure_filename(uploaded_file.filename)
            uploaded_file.save(os.path.join(
                app.config['UPLOAD_FOLDER'], filename))

            # Kaydedilen dosyayı işle
            img = load_img(os.path.join(
                app.config['UPLOAD_FOLDER'], filename), target_size=(64, 64))
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)

            predictions = model.predict(img)
            predicted_class = int(predictions[0][0])

            return render_template('result.html', predicted_class=predicted_class, uploaded_filename=filename)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=443)
