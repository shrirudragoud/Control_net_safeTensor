from flask import Flask, request, render_template, send_file, jsonify
from werkzeug.utils import secure_filename
import os
from pathlib import Path
from .main import FashionPipeline
from .config import Config
import logging

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload and output directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# Initialize pipeline
config = Config()
pipeline = FashionPipeline(config.get_config())

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        # Validate file type
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            return jsonify({'error': 'Invalid file type'}), 400
            
        # Save uploaded file
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)
        
        # Process image
        try:
            results = pipeline.process_image(
                image_path=input_path,
                prompt=request.form.get('prompt', None),
                num_images=1
            )
            
            # Save result
            output_filename = f"result_{filename}"
            output_path = os.path.join('outputs', output_filename)
            results[0].save(output_path)
            
            return jsonify({
                'success': True,
                'result_path': f'/result/{output_filename}'
            })
            
        except Exception as e:
            logging.error(f"Processing error: {str(e)}")
            return jsonify({'error': 'Error processing image'}), 500
            
    except Exception as e:
        logging.error(f"Upload error: {str(e)}")
        return jsonify({'error': 'Error uploading file'}), 500

@app.route('/result/<filename>')
def get_result(filename):
    return send_file(
        os.path.join('outputs', filename),
        mimetype='image/png'
    )

def run_server():
    app.run(host='0.0.0.0', port=8081, debug=True)

if __name__ == '__main__':
    run_server()