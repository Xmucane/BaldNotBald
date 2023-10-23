from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import os
import numpy as np
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Test'
model = load_model('modelim.h5', compile=True)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def classify_image():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            filename = secure_filename(uploaded_file.filename)
            uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # İşlem sonrası dosyaları kapat
            with Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename)) as img:
                img = img.resize((64, 64))
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)

                predictions = model.predict(img)
                predicted_class = int(predictions[0][0])

                return render_template('result.html', predicted_class=predicted_class, uploaded_filename=filename)

    return render_template('index.html')

if __name__ == '__main__':
    from gunicorn.app.wsgiapp import WSGIApplication

    gunicorn_opts = {
        'bind': '0.0.0.0:10000',  # Gunicorn'un çalıştığı adres ve port
        'workers': 2,  # İşçi sayısı
        'worker_class': 'gevent',  # İşçi sınıfı
        'worker_memory_limit': "200M",  # Bellek sınırlamasını azaltın
        'timeout': 60  # Zaman aşımı
    }

    WSGIApplication("%(prog)s [OPTIONS]").run()
