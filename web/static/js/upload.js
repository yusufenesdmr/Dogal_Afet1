/**
 * Upload Page JavaScript
 * Handles image upload, preview, and analysis
 */

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewArea = document.getElementById('previewArea');
const imagePreview = document.getElementById('imagePreview');
const analyzeBtn = document.getElementById('analyzeBtn');
const resetBtn = document.getElementById('resetBtn');
const loadingArea = document.getElementById('loadingArea');
const resultsArea = document.getElementById('resultsArea');

let selectedFile = null;

// ==== EVENT LISTENERS ====

// Click to upload
uploadArea.addEventListener('click', () => {
    fileInput.click();
});

// File selection
fileInput.addEventListener('change', (e) => {
    handleFileSelect(e.target.files[0]);
});

// Drag and drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    handleFileSelect(file);
});

// Analyze button
analyzeBtn.addEventListener('click', () => {
    if (selectedFile) {
        analyzeImage(selectedFile);
    }
});

// Reset button
resetBtn.addEventListener('click', () => {
    resetUpload();
});

// ==== FUNCTIONS ====

function handleFileSelect(file) {
    if (!file) return;

    // Validate file type
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/bmp', 'image/webp'];
    if (!validTypes.includes(file.type)) {
        alert('Lütfen geçerli bir görsel dosyası seçin (PNG, JPEG, GIF, BMP, WebP)');
        return;
    }

    // Validate file size (max 16MB)
    if (file.size > 16 * 1024 * 1024) {
        alert('Dosya boyutu 16MB\'dan küçük olmalıdır');
        return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        uploadArea.classList.add('hidden');
        previewArea.classList.remove('hidden');
        resultsArea.classList.add('hidden');
    };
    reader.readAsDataURL(file);
}

async function analyzeImage(file) {
    // Show loading
    previewArea.classList.add('hidden');
    resultsArea.classList.add('hidden');
    loadingArea.classList.remove('hidden');

    try {
        // Create form data
        const formData = new FormData();
        formData.append('image', file);

        // Send to API
        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Analiz başarısız oldu');
        }

        const result = await response.json();

        // Hide loading
        loadingArea.classList.add('hidden');
        previewArea.classList.remove('hidden');

        // Show results
        displayResults(result);

    } catch (error) {
        console.error('Error:', error);
        loadingArea.classList.add('hidden');
        previewArea.classList.remove('hidden');
        alert('Görsel analizi sırasında bir hata oluştu. Lütfen tekrar deneyin.');
    }
}

function displayResults(result) {
    resultsArea.classList.remove('hidden');

    const isSafe = !result.has_disaster;
    const cardClass = isSafe ? 'safe' : 'danger';
    const icon = isSafe ? '✅' : '⚠️';

    let html = `
        <div class="result-card ${cardClass}">
            <div class="flex-center gap-2 mb-3">
                <span style="font-size: 2.5rem;">${icon}</span>
                <div>
                    <h2>${result.disaster_type}</h2>
                    <p class="text-muted">Güven: %${(result.confidence * 100).toFixed(1)}</p>
                </div>
            </div>
            
            <div style="padding: 1rem; background: rgba(255,255,255,0.05); border-radius: 8px; margin-bottom: 1rem;">
                <p style="font-size: 1.1rem; line-height: 1.6;">${result.message}</p>
            </div>
            
            <div class="mb-3">
                <h3 style="margin-bottom: 1rem;">📊 Detaylı Analiz</h3>
                <div class="stats-grid">
    `;

    // Sort probabilities
    const probs = Object.entries(result.all_probabilities)
        .sort((a, b) => b[1] - a[1]);

    // Disaster icons
    const icons = {
        'Çığ': '🏔️',
        'Deprem': '🏚️',
        'Normal': '✅',
        'Sel': '🌊',
        'Yangın': '🔥'
    };

    probs.forEach(([disaster, prob]) => {
        const percentage = (prob * 100).toFixed(1);
        const barWidth = percentage;
        html += `
            <div class="stat-item" style="flex-direction: column; align-items: flex-start;">
                <div class="flex gap-2" style="width: 100%; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span>${icons[disaster]} ${disaster}</span>
                    <strong>${percentage}%</strong>
                </div>
                <div style="width: 100%; height: 6px; background: rgba(255,255,255,0.1); border-radius: 3px; overflow: hidden;">
                    <div style="width: ${barWidth}%; height: 100%; background: linear-gradient(90deg, #667eea, #764ba2); transition: width 0.5s ease;"></div>
                </div>
            </div>
        `;
    });

    html += `
                </div>
            </div>
            

    `;

    // Add recommendations if disaster
    if (!isSafe && result.recommendations) {
        html += `
            <div class="mt-4">
                <h3 style="margin-bottom: 1rem;">💡 Güvenlik Önerileri</h3>
                <div class="recommendations">
        `;

        result.recommendations.forEach((rec, index) => {
            html += `
                <div class="recommendation-item">
                    <strong>${index + 1}.</strong> ${rec}
                </div>
            `;
        });

        html += `
                </div>
            </div>
        `;
    }

    html += '</div>';

    resultsArea.innerHTML = html;

    // Scroll to results
    resultsArea.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    uploadArea.classList.remove('hidden');
    previewArea.classList.add('hidden');
    resultsArea.classList.add('hidden');
    loadingArea.classList.add('hidden');
}
