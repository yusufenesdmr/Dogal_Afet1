# 🌍 Disaster-Vision-AI: Afet Tespit ve Görsel Farkındalık Sistemi

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Flask](https://img.shields.io/badge/Flask-Web%20App-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

Bu proje, yapay zeka destekli bir **Doğal Afet Tespit ve Risk Analiz Platformudur**. Gelişmiş derin öğrenme mimarileri (**EfficientNetV2**) kullanılarak afet görselleri saniyeler içinde analiz edilir ve interaktif bir Türkiye haritası üzerinde bölgesel risk durumları görselleştirilir.

### 🎥 Proje Hakkında
Sistem, kullanıcı tarafından yüklenen fotoğrafları analiz ederek **Yangın, Sel, Deprem, Çığ** veya **Normal** durum olup olmadığını tespit eder. Aynı zamanda şehirlere özel risk haritaları oluşturarak görsel farkındalık sağlar.

---

## 🚀 Özellikler

*   **Yüksek Doğruluklu Yapay Zeka:** 5 farklı sınıfı %91.13 doğruluk oranıyla tespit eder.
*   **Transfer Learning Teknolojisi:** ImageNet ağırlıklarıyla eğitilmiş **EfficientNetV2-S** mimarisi.
*   **İnteraktif SVG Haritası:** Türkiye'nin tüm illerini (adalar dahil) kapsayan, veri odaklı dinamik risk haritası.
*   **Hızlı Analiz:** Yüklenen fotoğrafları milisaniyeler içinde işleyen optimize edilmiş inference motoru.
*   **Modern Arayüz:** Kullanıcı dostu, responsive ve şık web arayüzü.

## 🛠️ Kurulum ve Çalıştırma

Projeyi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin:

1.  **Projeyi İndirin:**
    ```bash
    git clone https://github.com/KULLANICI_ADI/Disaster-Vision-AI.git
    cd Disaster-Vision-AI
    ```

2.  **Gerekli Kütüphaneleri Yükleyin:**
    ```bash
    pip install flask torch torchvision pillow numpy scikit-learn matplotlib seaborn
    ```

3.  **Uygulamayı Başlatın:**
    ```bash
    cd web
    python app.py
    ```

4.  **Tarayıcıda Açın:**
    `http://localhost:5000` adresine gidin.

## 📊 Model Performansı

Modelimiz zorlu koşullarda test edilmiştir. Detaylı eğitim grafikleri `model/results` klasöründedir.

| Metrik | Değer |
|:---:|:---:|
| **Model** | EfficientNetV2-S |
| **Accuracy** | %91.13 |
| **Loss** | 0.24 |
| **Epoch** | 25 (Early Stopping) |

---
---

# 🌍 Disaster-Vision-AI: Disaster Detection & Awareness System

**Disaster-Vision-AI** is a deep learning-based platform designed to detect natural disasters from images and visualize regional risks on an interactive map.

## 🚀 Features

*   **Advanced AI Model:** Detects 5 classes (**Fire, Flood, Earthquake, Avalanche, Normal**) with **91.13% accuracy**.
*   **Architecture:** Powered by **EfficientNetV2-S** using Transfer Learning.
*   **Interactive Map:** Dynamic SVG map of Turkey reflecting real-time disaster risks per city.
*   **Real-time Analysis:** Optimized pipeline for instant image classification.

## 🛠️ Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/USERNAME/Disaster-Vision-AI.git
    cd Disaster-Vision-AI
    ```

2.  **Install Dependencies:**
    ```bash
    pip install flask torch torchvision pillow numpy scikit-learn matplotlib seaborn
    ```

3.  **Run the App:**
    ```bash
    cd web
    python app.py
    ```

4.  **Access:**
    Open `http://localhost:5000` in your browser.

## 📂 Project Structure

*   `web/`: Flask application and interface codes.
*   `model/`: Training scripts and performance graphs.
*   `database/`: Dataset structure (Train/Test).

---
*Developed using Python & PyTorch.*
