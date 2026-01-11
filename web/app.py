"""
Flask Web Application for Disaster Detection
Features: Image upload & analysis, Interactive Turkey map, City statistics
"""

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import json
import random
from model_handler import DisasterDetector

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Initialize model
MODEL_PATH = os.path.join('..', 'model', 'model_checkpoint', 'best_model.pth')
detector = DisasterDetector(MODEL_PATH)

# Load static data
with open('static/data/city_stats.json', 'r', encoding='utf-8') as f:
    CITY_STATS = json.load(f)

with open('static/data/recommendations.json', 'r', encoding='utf-8') as f:
    RECOMMENDATIONS = json.load(f)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """Main page - Image upload and analysis"""
    return render_template('index.html')

@app.route('/map')
def map_page():
    """Interactive Turkey map page"""
    return render_template('map.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """
    Analyze uploaded image for disasters
    
    Returns:
        JSON: Prediction results, probabilities, and recommendations
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            # Read image bytes
            image_bytes = file.read()
            
            # Get prediction from model
            prediction = detector.predict(image_bytes)
            
            # Add message
            prediction['message'] = detector.get_disaster_message(prediction)
            
            # Add recommendations if disaster detected
            if prediction['has_disaster']:
                disaster_type = prediction['disaster_type']
                if disaster_type in RECOMMENDATIONS:
                    recs = RECOMMENDATIONS[disaster_type]
                    prediction['recommendations'] = random.sample(recs, min(3, len(recs)))
            
            prediction['status'] = 'success'
            return jsonify(prediction)
        
        return jsonify({'error': 'Invalid file type'}), 400
    
    except Exception as e:
        print(f"Error in analyze_image: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/city-stats', methods=['GET'])
def get_city_stats():
    """
    Get all city disaster statistics
    
    Returns:
        JSON: All 81 cities with disaster percentages
    """
    return jsonify(CITY_STATS)

@app.route('/api/city-stats/<city>', methods=['GET'])
def get_city_stat(city):
    """
    Get specific city disaster statistics
    
    Args:
        city: City name in Turkish
        
    Returns:
        JSON: City disaster percentages
    """
    if city in CITY_STATS:
        return jsonify({
            'city': city,
            'stats': CITY_STATS[city]
        })
    return jsonify({'error': 'City not found'}), 404

@app.route('/api/recommendations/<disaster_type>', methods=['GET'])
def get_recommendation(disaster_type):
    """
    Get random recommendation for specific disaster type
    
    Args:
        disaster_type: Type of disaster (Sel, Yangın, Deprem, Çığ)
        
    Returns:
        JSON: Random recommendation
    """
    # Map English to Turkish names if needed
    disaster_map = {
        'Sel': 'Sel',
        'Yangın': 'Yangın',
        'Yangin': 'Yangın',
        'Deprem': 'Deprem',
        'Çığ': 'Çığ',
        'Cig': 'Çığ'
    }
    
    disaster = disaster_map.get(disaster_type, disaster_type)
    
    if disaster in RECOMMENDATIONS:
        rec = random.choice(RECOMMENDATIONS[disaster])
        return jsonify({
            'disaster': disaster,
            'recommendation': rec
        })
    
    return jsonify({'error': 'Disaster type not found'}), 404

@app.route('/api/city-recommendations/<city>', methods=['GET'])
def get_city_recommendations(city):
    """
    Get recommendations based on city's highest risk disaster
    
    Args:
        city: City name
        
    Returns:
        JSON: Top disaster type and recommendations
    """
    if city not in CITY_STATS:
        return jsonify({'error': 'City not found'}), 404
    
    stats = CITY_STATS[city]
    # Find highest risk disaster
    top_disaster = max(stats, key=stats.get)
    top_percentage = stats[top_disaster]
    
    # Get multiple recommendations
    recs = RECOMMENDATIONS.get(top_disaster, [])
    selected_recs = random.sample(recs, min(5, len(recs)))
    
    return jsonify({
        'city': city,
        'top_disaster': top_disaster,
        'risk_percentage': top_percentage,
        'recommendations': selected_recs
    })

if __name__ == '__main__':
    # Create uploads folder if not exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    print("\n" + "="*70)
    print("AFET TESPİT WEB UYGULAMASI")
    print("="*70)
    print(f"✓ Model yüklendi: {MODEL_PATH}")
    print(f"✓ Şehir verileri:  {len(CITY_STATS)} il")
    print(f"✓ Tavsiye sistemi hazır")
    print("="*70)
    print("\n🌐 Sunucu başlatılıyor: http://localhost:5000")
    print("   • Ana Sayfa: http://localhost:5000")
    print("   • Harita:    http://localhost:5000/map")
    print("\nÇıkmak için Ctrl+C")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
