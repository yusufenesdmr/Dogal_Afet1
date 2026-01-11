"""
Model Handler - EfficientNetV2 Disaster Detection
Handles model loading and inference for disaster classification
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import time
import io

class DisasterDetector:
    """Disaster detection model handler"""
    
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = ['Çığ', 'Deprem', 'Normal', 'Sel', 'Yangın']
        self.model = self._load_model(model_path)
        self.transform = self._get_transform()
        print(f"✓ Model loaded successfully on {self.device}")
    
    def _load_model(self, path):
        """Load EfficientNetV2 model with trained weights"""
        model = models.efficientnet_v2_s(weights=None)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(num_features, 5)
        )
        
        checkpoint = torch.load(path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model
    
    def _get_transform(self):
        """Get image preprocessing transformation"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_bytes):
        """
        Predict disaster type from image bytes
        
        Args:
            image_bytes: Image file bytes
            
        Returns:
            dict: Prediction results with probabilities and metadata
        """
        start_time = time.time()
        
        # Load and preprocess image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            confidence, predicted_class = torch.max(probabilities, 0)
        
        # Convert to percentages
        all_probs = {
            self.classes[i]: float(probabilities[i])
            for i in range(len(self.classes))
        }
        
        predicted_disaster = self.classes[predicted_class.item()]
        confidence_score = float(confidence.item())
        
        # Determine if it's a disaster
        is_disaster = predicted_disaster != 'Normal'
        
        # Create response
        result = {
            'has_disaster': is_disaster,
            'disaster_type': predicted_disaster,
            'confidence': confidence_score,
            'all_probabilities': all_probs,
            'analysis_time': round(time.time() - start_time, 2)
        }
        
        return result
    
    def get_disaster_message(self, prediction):
        """Generate human-readable message from prediction"""
        disaster_type = prediction['disaster_type']
        confidence = prediction['confidence']
        
        if not prediction['has_disaster']:
            return "✅ Güvenli - Bu görselde afet belirtisi tespit edilmedi."
        
        messages = {
            'Yangın': f'🔥 Yangın tespit edildi! Lütfen dikkatli olun ve gerekirse yetkilileri arayın.',
            'Sel': f'🌊 Sel/Su baskını tespit edildi! Güvenli bir yere çıkın.',
            'Deprem': f'🏚️ Deprem hasarı tespit edildi! Güvenli bölgede kalın.',
            'Çığ': f'🏔️ Çığ tespit edildi! Dağlık bölgelerden uzak durun.'
        }
        
        return messages.get(disaster_type, 'Afet tespit edildi.')
