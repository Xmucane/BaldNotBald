from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import os
import numpy as np

# Diğer kodlar burada

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

# Diğer kodlar burada

if __name__ == '__main__':
    app.run(debug=True)
