import os
import uuid
import numpy as np
import tempfile
from flask import Flask, render_template, Response, send_from_directory, jsonify, request
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS
from PIL import Image
import torch

app = Flask(__name__)
AUDIO_DIR = os.path.join(tempfile.gettempdir(), "audio_files")
os.makedirs(AUDIO_DIR, exist_ok=True)

# Load BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image):
    pil_image = Image.fromarray(image)
    inputs = processor(pil_image, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs)
    return processor.decode(output[0], skip_special_tokens=True)

def text_to_speech(text):
    filename = f"speech_{uuid.uuid4()}.mp3"
    filepath = os.path.join(AUDIO_DIR, filename)
    try:
        tts = gTTS(text=text, lang='en')
        tts.save(filepath)
        return filename
    except Exception as e:
        print("TTS Error:", e)
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    if 'frame' not in request.files:
        return jsonify({'error': 'No image received'}), 400

    image_file = request.files['frame']
    image = Image.open(image_file.stream).convert("RGB")
    np_image = np.array(image)

    caption = generate_caption(np_image)
    audio_file = text_to_speech(caption)

    return jsonify({
        "caption": caption,
        "image_url": "",  # No saved image
        "audio_url": f"/audio/{audio_file}" if audio_file else None
    })

@app.route('/audio/<filename>')
def audio_file(filename):
    return send_from_directory(AUDIO_DIR, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
