"""
Doğal Afet Tespit Modeli - Kapsamlı Eğitim Scripti
Model: EfficientNetV2-S with Transfer Learning
Sınıflar: Çığ, Deprem, Yangın, Sel, Normal
"""

import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import warnings
warnings.filterwarnings('ignore')
# Rastgeleliği sabitle (reproducibility için)
torch.manual_seed(42)
np.random.seed(42)

# ==================== KONFİGÜRASYON ====================
class Config:
    # Klasörler
    DATA_DIR = os.path.join('..', 'database', 'train')
    RESULTS_DIR = 'results'
    CHECKPOINT_DIR = 'model_checkpoint'
    
    # Model hiperparametreleri
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.01
    
    # Early stopping
    PATIENCE = 15
    
    # Data split
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # Model
    NUM_CLASSES = 5
    IMG_SIZE = 224
    DROPOUT_RATE = 0.3
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Sınıf isimleri
    CLASS_NAMES = ['cig1', 'deprem1', 'normal1', 'sel1', 'yangın1']
    CLASS_NAMES_TR = ['Çığ', 'Deprem', 'Normal', 'Sel', 'Yangın']

config = Config()

# Klasör oluşturma
os.makedirs(config.RESULTS_DIR, exist_ok=True)
os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

print("=" * 70)
print("DOĞAL AFET TESPİT MODELİ EĞİTİMİ")
print("=" * 70)
print(f"Device: {config.DEVICE}")
if config.DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print("=" * 70)

# ==================== VERİ HAZIRLAMA ====================

# Data augmentation (sadece eğitim için)
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Validation/Test transformations (sadece normalizasyon)
val_test_transforms = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Tüm veri setini yükle
full_dataset = datasets.ImageFolder(config.DATA_DIR)
print(f"\n✓ Toplam {len(full_dataset)} görsel yüklendi")
print(f"✓ Sınıflar: {full_dataset.classes}")

# Sınıf dağılımını analiz et
class_counts = {}
for idx in range(len(full_dataset)):
    label = full_dataset.targets[idx]
    class_name = full_dataset.classes[label]
    class_counts[class_name] = class_counts.get(class_name, 0) + 1

print("\nSınıf Dağılımı:")
print("-" * 40)
for class_name, count in class_counts.items():
    print(f"  {class_name:15}: {count:5} görsel")
print("-" * 40)

# Stratified train/val/test split
indices = list(range(len(full_dataset)))
labels = full_dataset.targets

# Train + Val ve Test ayırma
train_val_idx, test_idx = train_test_split(
    indices, test_size=config.TEST_RATIO, stratify=labels, random_state=42
)

# Train ve Val ayırma
train_val_labels = [labels[i] for i in train_val_idx]
train_idx, val_idx = train_test_split(
    train_val_idx, 
    test_size=config.VAL_RATIO / (config.TRAIN_RATIO + config.VAL_RATIO),
    stratify=train_val_labels,
    random_state=42
)

print(f"\n✓ Train: {len(train_idx)} görsel ({len(train_idx)/len(full_dataset)*100:.1f}%)")
print(f"✓ Val:   {len(val_idx)} görsel ({len(val_idx)/len(full_dataset)*100:.1f}%)")
print(f"✓ Test:  {len(test_idx)} görsel ({len(test_idx)/len(full_dataset)*100:.1f}%)")

# Dataset'leri oluştur
train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(full_dataset, val_idx)
test_dataset = Subset(full_dataset, test_idx)

# Transform'ları uygula
train_dataset.dataset.transform = train_transforms
val_dataset.dataset.transform = val_test_transforms
test_dataset.dataset.transform = val_test_transforms

# DataLoader'ları oluştur
train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

print(f"\n✓ DataLoader'lar oluşturuldu (batch_size={config.BATCH_SIZE})")

# ==================== MODEL OLUŞTURMA ====================

def create_model():
    """EfficientNetV2-S modelini oluştur"""
    # Pretrained model yükle
    model = models.efficientnet_v2_s(weights='DEFAULT')
    
    # Son fully connected katmanını değiştir
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=config.DROPOUT_RATE, inplace=True),
        nn.Linear(num_features, config.NUM_CLASSES)
    )
    
    return model

model = create_model()
model = model.to(config.DEVICE)

print(f"\n✓ Model oluşturuldu: EfficientNetV2-S")
print(f"✓ Dropout rate: {config.DROPOUT_RATE}")
print(f"✓ Çıkış sınıf sayısı: {config.NUM_CLASSES}")

# ==================== LOSS & OPTIMIZER ====================

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

print(f"\n✓ Optimizer: AdamW (lr={config.LEARNING_RATE}, weight_decay={config.WEIGHT_DECAY})")
print(f"✓ Loss: CrossEntropyLoss")
print(f"✓ LR Scheduler: ReduceLROnPlateau")

# ==================== EĞİTİM FONKSİYONLARI ====================

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Bir epoch eğitimi"""
    model.train()
    running_loss = 0.0
    running_corrects = 0
    
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    
    return epoch_loss, epoch_acc.item()

def validate(model, dataloader, criterion, device):
    """Validation/Test değerlendirmesi"""
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    
    return epoch_loss, epoch_acc.item()

# ==================== EĞİTİM LOOPU ====================

print("\n" + "=" * 70)
print("EĞİTİM BAŞLIYOR")
print("=" * 70)

# Eğitim geçmişi
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': [],
    'lr': []
}

best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0
best_loss = float('inf')
patience_counter = 0

start_time = time.time()

# Log dosyası
log_file = open(os.path.join(config.RESULTS_DIR, 'training_log.txt'), 'w')
log_file.write("Epoch,Train_Loss,Train_Acc,Val_Loss,Val_Acc,LR\n")

for epoch in range(config.NUM_EPOCHS):
    epoch_start = time.time()
    
    # Training
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
    
    # Validation
    val_loss, val_acc = validate(model, val_loader, criterion, config.DEVICE)
    
    # LR Scheduler step
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    # Geçmişi kaydet
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['lr'].append(current_lr)
    
    # Log
    epoch_time = time.time() - epoch_start
    print(f"Epoch [{epoch+1:3d}/{config.NUM_EPOCHS}] | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
          f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")
    
    log_file.write(f"{epoch+1},{train_loss:.6f},{train_acc:.6f},{val_loss:.6f},{val_acc:.6f},{current_lr:.8f}\n")
    log_file.flush()
    
    # En iyi modeli kaydet
    if val_loss < best_loss:
        best_loss = val_loss
        best_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
        }, os.path.join(config.CHECKPOINT_DIR, 'best_model.pth'))
        patience_counter = 0
        print(f"  ✓ En iyi model kaydedildi (Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f})")
    else:
        patience_counter += 1
    
    # Early stopping
    if patience_counter >= config.PATIENCE:
        print(f"\n⚠ Early stopping triggered! ({config.PATIENCE} epoch boyunca iyileşme yok)")
        break

log_file.close()
total_time = time.time() - start_time

print("\n" + "=" * 70)
print("EĞİTİM TAMAMLANDI!")
print("=" * 70)
print(f"Toplam Süre: {total_time/60:.2f} dakika")
print(f"En İyi Val Accuracy: {best_acc:.4f}")
print(f"En İyi Val Loss: {best_loss:.4f}")

# En iyi modeli yükle
model.load_state_dict(best_model_wts)

# ==================== TEST DEĞERLENDİRMESİ ====================

print("\n" + "=" * 70)
print("TEST SETİ DEĞERLENDİRMESİ")
print("=" * 70)

test_loss, test_acc = validate(model, test_loader, criterion, config.DEVICE)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Tahminleri topla
all_preds = []
all_labels = []
all_probs = []

model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(config.DEVICE)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# ==================== GÖRSELLEŞTİRMELER ====================

print("\nGörselleştirmeler oluşturuluyor...")

# 1. CONFUSION MATRIX (HAM)
plt.figure(figsize=(10, 8))
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=config.CLASS_NAMES_TR, 
            yticklabels=config.CLASS_NAMES_TR,
            cbar_kws={'label': 'Sayı'})
plt.title('Karmaşıklık Matrisi (Confusion Matrix)', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Gerçek Etiket', fontsize=12)
plt.xlabel('Tahmin Edilen Etiket', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(config.RESULTS_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Karmaşıklık matrisi kaydedildi")

# 2. CONFUSION MATRIX (NORMALİZE)
plt.figure(figsize=(10, 8))
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens',
            xticklabels=config.CLASS_NAMES_TR,
            yticklabels=config.CLASS_NAMES_TR,
            cbar_kws={'label': 'Oran'})
plt.title('Normalize Karmaşıklık Matrisi', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Gerçek Etiket', fontsize=12)
plt.xlabel('Tahmin Edilen Etiket', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(config.RESULTS_DIR, 'confusion_matrix_normalized.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Normalize karmaşıklık matrisi kaydedildi")

# 3. EĞİTİM GRAFİKLERİ
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Loss
axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2, marker='o', markersize=3)
axes[0, 0].plot(history['val_loss'], label='Validation Loss', linewidth=2, marker='s', markersize=3)
axes[0, 0].set_xlabel('Epoch', fontsize=12)
axes[0, 0].set_ylabel('Loss', fontsize=12)
axes[0, 0].set_title('Eğitim ve Doğrulama Kayıp (Loss)', fontsize=14, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# Accuracy
axes[0, 1].plot(history['train_acc'], label='Train Accuracy', linewidth=2, marker='o', markersize=3)
axes[0, 1].plot(history['val_acc'], label='Validation Accuracy', linewidth=2, marker='s', markersize=3)
axes[0, 1].set_xlabel('Epoch', fontsize=12)
axes[0, 1].set_ylabel('Accuracy', fontsize=12)
axes[0, 1].set_title('Eğitim ve Doğrulama Doğruluk (Accuracy)', fontsize=14, fontweight='bold')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3)

# Learning Rate
axes[1, 0].plot(history['lr'], linewidth=2, marker='o', markersize=3, color='purple')
axes[1, 0].set_xlabel('Epoch', fontsize=12)
axes[1, 0].set_ylabel('Learning Rate', fontsize=12)
axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
axes[1, 0].set_yscale('log')
axes[1, 0].grid(True, alpha=0.3)

# Train-Val Gap (Overfitting göstergesi)
loss_gap = [t - v for t, v in zip(history['train_loss'], history['val_loss'])]
axes[1, 1].plot(loss_gap, linewidth=2, marker='o', markersize=3, color='red')
axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[1, 1].set_xlabel('Epoch', fontsize=12)
axes[1, 1].set_ylabel('Train Loss - Val Loss', fontsize=12)
axes[1, 1].set_title('Overfitting Göstergesi (Loss Gap)', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(config.RESULTS_DIR, 'training_curves.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Eğitim grafikleri kaydedildi")

# 4. SINIF DAĞILIMI
plt.figure(figsize=(12, 6))
class_counts_sorted = {config.CLASS_NAMES_TR[i]: class_counts[config.CLASS_NAMES[i]] 
                       for i in range(len(config.CLASS_NAMES))}
bars = plt.bar(class_counts_sorted.keys(), class_counts_sorted.values(), 
               color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'],
               edgecolor='black', linewidth=1.5)
plt.xlabel('Sınıf', fontsize=12)
plt.ylabel('Görsel Sayısı', fontsize=12)
plt.title('Veri Seti Sınıf Dağılımı', fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=0, fontsize=11)

# Bar üzerinde sayıları göster
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(config.RESULTS_DIR, 'class_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Sınıf dağılımı grafiği kaydedildi")

# 5. ROC EĞRİLERİ
plt.figure(figsize=(12, 10))

# One-vs-Rest için binary labels
y_test_binary = label_binarize(all_labels, classes=range(config.NUM_CLASSES))

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
for i in range(config.NUM_CLASSES):
    fpr, tpr, _ = roc_curve(y_test_binary[:, i], all_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=colors[i], lw=2.5, 
             label=f'{config.CLASS_NAMES_TR[i]} (AUC = {roc_auc:.3f})')

# Diagonal line
plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Rastgele (AUC = 0.5)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Eğrileri (One-vs-Rest)', fontsize=16, fontweight='bold', pad=20)
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(config.RESULTS_DIR, 'roc_curves.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ ROC eğrileri kaydedildi")

# 6. CLASSIFICATION REPORT
report = classification_report(all_labels, all_preds, 
                               target_names=config.CLASS_NAMES_TR,
                               digits=4)
with open(os.path.join(config.RESULTS_DIR, 'classification_report.txt'), 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("SINIF BAZLI PERFORMANS RAPORU\n")
    f.write("=" * 70 + "\n\n")
    f.write(report)
    f.write("\n\n" + "=" * 70 + "\n")
    f.write(f"Test Accuracy: {test_acc:.4f}\n")
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write("=" * 70 + "\n")
print("✓ Classification report kaydedildi")

print(report)

# 7. ÖRNEK TAHMİNLER
fig, axes = plt.subplots(3, 5, figsize=(18, 12))
axes = axes.ravel()

# Rastgele 15 örnek seç
np.random.seed(42)
random_indices = np.random.choice(len(test_dataset), size=15, replace=False)

# Denormalizasyon için
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

model.eval()
with torch.no_grad():
    for idx, sample_idx in enumerate(random_indices):
        img, label = test_dataset[sample_idx]
        img_display = inv_normalize(img).permute(1, 2, 0).numpy()
        img_display = np.clip(img_display, 0, 1)
        
        img_batch = img.unsqueeze(0).to(config.DEVICE)
        output = model(img_batch)
        prob = torch.softmax(output, dim=1)
        pred_class = torch.argmax(prob, dim=1).item()
        confidence = prob[0, pred_class].item()
        
        true_label = config.CLASS_NAMES_TR[label]
        pred_label = config.CLASS_NAMES_TR[pred_class]
        
        axes[idx].imshow(img_display)
        axes[idx].axis('off')
        
        color = 'green' if pred_class == label else 'red'
        title = f"Gerçek: {true_label}\nTahmin: {pred_label}\n({confidence*100:.1f}%)"
        axes[idx].set_title(title, fontsize=9, color=color, fontweight='bold')

plt.suptitle('Örnek Tahminler (Yeşil: Doğru, Kırmızı: Yanlış)', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(config.RESULTS_DIR, 'sample_predictions.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Örnek tahminler kaydedildi")

# ==================== SONUÇ ÖZETİ ====================

print("\n" + "=" * 70)
print("TÜM SONUÇLAR BAŞARIYLA KAYDEDİLDİ!")
print("=" * 70)
print(f"\n📁 Model Checkpoint: {config.CHECKPOINT_DIR}/best_model.pth")
print(f"📊 Sonuçlar Klasörü: {config.RESULTS_DIR}/")
print("\nOluşturulan Dosyalar:")
print("  ✓ confusion_matrix.png - Karmaşıklık matrisi")
print("  ✓ confusion_matrix_normalized.png - Normalize karmaşıklık matrisi")
print("  ✓ training_curves.png - Eğitim grafikleri")
print("  ✓ class_distribution.png - Sınıf dağılımı")
print("  ✓ roc_curves.png - ROC eğrileri")
print("  ✓ sample_predictions.png - Örnek tahminler")
print("  ✓ classification_report.txt - Detaylı performans raporu")
print("  ✓ training_log.txt - Epoch bazında eğitim logu")
print("\n" + "=" * 70)
print(f"🎯 FINAL TEST ACCURACY: {test_acc*100:.2f}%")
print("=" * 70)
