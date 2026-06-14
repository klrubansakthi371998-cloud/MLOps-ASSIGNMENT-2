import os
import joblib
import numpy as np
from flask import Flask, request, render_template_string
from PIL import Image

app = Flask(__name__)

# Locate the model file in the same directory as this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'savedmodel.pth')

# Load the model
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# HTML interface for the web app
HTML_TEMPLATE = '''
<!doctype html>
<html>
  <body style="font-family: sans-serif; text-align: center; margin-top: 50px;">
    <h2>Olivetti Face Dataset Classifier</h2>
    <form method="POST" enctype="multipart/form-data">
      <input type="file" name="file" accept="image/*" required>
      <input type="submit" value="Run Prediction Inference">
    </form>
    {% if prediction is not none %}
      <h3 style="color: blue;">Predicted Subject ID: {{ prediction }}</h3>
    {% endif %}
  </body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            try:
                # 1. Open and convert to Grayscale ('L')
                img = Image.open(file.stream).convert('L')
                
                # 2. Force resize to 64x64
                img = img.resize((64, 64))
                
                # 3. Convert to array and ensure it's (1, 4096) shape
                img_array = np.array(img).flatten().reshape(1, -1)
                
                # 4. Normalize (if your model was trained on 0-1 values)
                img_array = img_array / 255.0 
                
                # 5. Predict
                prediction = int(model.predict(img_array)[0])
                print(f"Prediction successful: {prediction}")
            except Exception as e:
                print(f"CRITICAL ERROR in processing: {e}")
                return f"Error: {str(e)}", 500
                
    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == '__main__':
    # Running on 0.0.0.0 is required for Docker containers to be accessible
    app.run(host='0.0.0.0', port=5000)