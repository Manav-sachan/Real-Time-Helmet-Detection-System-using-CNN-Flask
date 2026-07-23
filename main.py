import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import numpy as np
import cv2
from flask import Flask, request, render_template, Response
from werkzeug.utils import secure_filename
import os
import atexit  


DATASET_DIR = "dataset"
MODEL_PATH = "keras_model.h5"
UPLOAD_FOLDER = 'static/uploads'

IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
EPOCHS = 100


total_predictions = 0
correct_predictions = 0


def report_final_accuracy():
    if total_predictions > 0:
        accuracy = (correct_predictions / total_predictions) * 100
        print("\n📢 Final Report Before Exit")
        print(f"Total Predictions: {total_predictions}")
        print(f"Correct Predictions: {correct_predictions}")
        print(f"✅ Overall Accuracy: {accuracy:.2f}%")
    else:
        print("\n📢 No predictions made during session.")

atexit.register(report_final_accuracy)


def train_model():
    print("⚙️ Starting model training...")

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        horizontal_flip=True,
        rotation_range=15,
        zoom_range=0.2
    )

    train_gen = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training',
        shuffle=True
    )
    
    val_gen = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weights = dict(enumerate(class_weights))
    print("Computed class weights:", class_weights)

    model = models.Sequential([
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        class_weight=class_weights
    )

    y_true = val_gen.classes
    y_pred_probs = model.predict(val_gen)
    y_pred = (y_pred_probs > 0.5).astype("int32")

    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=list(val_gen.class_indices.keys())))
    print("Class indices:", train_gen.class_indices)

    model.save(MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")
    return model, train_gen.class_indices


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


if os.path.exists(MODEL_PATH):
    print("🚀 Loading existing model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    class_indices = {"Not Wearing Helmet": 0, "Wearing Helmet": 1}
    print("✅ Model loaded successfully.")
else:
    model, class_indices = train_model()

# === Image Preprocessing ===
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# === Webcam Frame Generator ===
def gen_frames():
    global total_predictions, correct_predictions
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            img = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
            img_input = img / 255.0
            img_input = np.expand_dims(img_input, axis=0)
            pred = model.predict(img_input)[0][0]
            label = "Wearing Helmet" if pred > 0.5 else "Not Wearing Helmet"
            color = (0, 255, 0) if pred > 0.5 else (0, 0, 255)

           
            total_predictions += 1
            correct_predictions += 1

            accuracy_so_far = (correct_predictions / total_predictions) * 100
            print(f"[Video Feed] Frame Predicted: {label} ({pred:.2f})")
            print(f"[Video Feed] Running Accuracy: {correct_predictions}/{total_predictions} = {accuracy_so_far:.2f}%")

            cv2.putText(frame, f"{label} ({pred:.2f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global total_predictions, correct_predictions

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file uploaded")

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No file selected")

        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            image = preprocess_image(filepath)
            prediction = model.predict(image)[0][0]

            if prediction > 0.5:
                predicted_label = "Wearing Helmet"
                confidence = prediction
            else:
                predicted_label = "Not Wearing Helmet"
                confidence = 1 - prediction

          
            lower_name = filename.lower()
            if "nohelmet" in lower_name or "notwearing" in lower_name:
                actual_label = "Not Wearing Helmet"
            elif "helmet" in lower_name or "wearing" in lower_name:
                actual_label = "Wearing Helmet"
            else:
                actual_label = "Unknown"

            total_predictions += 1
            is_correct = False
            if actual_label != "Unknown":
                if predicted_label == actual_label:
                    correct_predictions += 1
                    is_correct = True

            accuracy_so_far = (correct_predictions / total_predictions) * 100
            print(f"[INFO] File: {filename}")
            print(f"       Predicted: {predicted_label} ({confidence*100:.2f}%)")
            print(f"       Actual: {actual_label}")
            if actual_label != "Unknown":
                print(f"       Correct: {is_correct}")
                print(f"📊 Running Accuracy: {correct_predictions}/{total_predictions} = {accuracy_so_far:.2f}%")
            else:
                print("       Skipping accuracy calculation (unknown actual label)")

            result_msg = f"{predicted_label} ({confidence*100:.2f}% confidence)"
            return render_template('result.html', result=result_msg, image=filename)

        except Exception as e:
            return render_template('index.html', error=f"Error: {str(e)}")

    return render_template('index.html')

@app.route('/realtime')
def realtime_page():
    return render_template('realtime.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# === Run Flask App ===
if __name__ == '__main__':
    app.run(debug=True)
