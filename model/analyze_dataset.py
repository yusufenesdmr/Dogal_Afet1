import os

folders = ['cig1', 'deprem1', 'normal1', 'sel1', 'yangın1']
base_path = 'veritabanı1'

print("Veri Seti Sınıf Dağılımı:")
print("="*50)
total = 0
for folder in folders:
    folder_path = os.path.join(base_path, folder)
    count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
    total += count
    print(f"{folder:15}: {count:5} görsel")
print("="*50)
print(f"{'TOPLAM':15}: {total:5} görsel")
