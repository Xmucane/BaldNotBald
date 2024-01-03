from flask import Flask, render_template, Response, jsonify
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
from torchvision.models import vgg16
import cv2

app = Flask(__name__)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class YourModelClass:
    def __init__(self, model_path='modelim.pth'):
        self.model = MyModel()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def predict(self, img):
        transform = transforms.Compose([
            transforms.Resize((80, 80)),
            transforms.ToTensor(),
        ])
        img = transform(img)
        img = img.unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(img)
            _, predicted_class = torch.max(outputs, 1)
        return predicted_class.item()

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.base_model = vgg16(pretrained=True)
        in_features = self.base_model.classifier[6].in_features
        self.base_model.classifier[6] = nn.Linear(in_features, 3)

    def forward(self, x):
        return self.base_model(x)

model_instance = YourModelClass()
frame_rate = 60
resolution = (640, 480)

def is_camera_working():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return False
    cap.release()
    return True

def set_camera_properties(cap, resolution, frame_rate):
    cap.set(3, resolution[0])
    cap.set(4, resolution[1])
    cap.set(5, frame_rate)

def detect_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return frame

# Eklendi: "/open_camera" endpoint'i
@app.route('/open_camera')
def open_camera():
    # Burada kamerayı açacak işlemleri gerçekleştirin
    # Örneğin: model_instance.start_camera() veya başka bir açma işlemi

    # Açma işlemi başarılıysa JSON yanıtı gönder
    return jsonify({'status': 'success'})

# Eklendi: "/get_camera_status" endpoint'i
@app.route('/get_camera_status')
def get_camera_status():
    camera_status = is_camera_working()
    return jsonify({'status': 'Çalışıyor' if camera_status else 'Beklemede'})

@app.route('/')
def index():
    camera_status = is_camera_working()
    return render_template('index_realtime.html', camera_status=camera_status, resolution=resolution, frame_rate=frame_rate)

def gen_frames():
    cap = cv2.VideoCapture(0)
    set_camera_properties(cap, resolution, frame_rate)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            predicted_class = model_instance.predict(pil_image)
            frame = detect_face(frame)
            font = cv2.FONT_HERSHEY_SIMPLEX
            label = "Kel" if predicted_class == 0 else ("Kismi Kel" if predicted_class == 1 else "Kel Degil")
            cv2.putText(frame, f'Sonuc: {label}', (10, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=False)
