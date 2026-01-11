import geopandas as gpd
import matplotlib.pyplot as plt
import os

def generate_turkey_map():
    print("1. Kütüphaneler yüklendi...")
    
    # URL Listesi
    urls = [
        "https://raw.githubusercontent.com/cihadturhan/turkey-maps/master/provinces/tr-provinces.json",
        "https://raw.githubusercontent.com/izzetkalic/geojsons-of-turkey/master/geojsons/turkey-admin-level-4.geojson",
        "https://raw.githubusercontent.com/alpers/Turkey-Maps-GeoJSON/master/tr-cities.json"
    ]
    
    gdf = None
    for url in urls:
        print(f"2. GeoJSON indiriliyor: {url}")
        try:
            gdf = gpd.read_file(url)
            print(f"   ✓ Veri indirildi. İl Sayısı: {len(gdf)}")
            break
        except Exception as e:
            print(f"   X Hata: {str(e)}")
            continue
            
    if gdf is None: return

    # Kolon bulma
    name_col = 'name'
    for col in ['name', 'NAME', 'il_adi', 'ADM1_TR', 'shapeName']:
        if col in gdf.columns:
            name_col = col
            break

    # Projeksiyon
    gdf = gdf.to_crs(epsg=3857)

    # DEBUG: ID Listesi
    generated_ids = []
    
    # ÇİZİM
    print("4. Harita çiziliyor (Geliştirilmiş Görünüm)...")
    
    fig, ax = plt.subplots(figsize=(24, 12)) 
    ax.set_axis_off()
    
    for idx, row in gdf.iterrows():
        city_name = row[name_col]
        
        if isinstance(city_name, str):
            safe_id = city_name
        else:
            safe_id = f"city_{idx}"

        # Türkçe karakterleri değiştir
        safe_id = safe_id.replace('İ', 'I').replace('ı', 'i').replace('ş', 's').replace('ğ', 'g').replace('ü', 'u').replace('ö', 'o').replace('ç', 'c')
        safe_id = safe_id.replace('Ş', 'S').replace('Ğ', 'G').replace('Ü', 'U').replace('Ö', 'O').replace('Ç', 'C')
        
        # Boşlukları sil
        safe_id = safe_id.replace(' ', '')
        
        generated_ids.append((city_name, safe_id))
        
        gpd.GeoSeries(row.geometry).plot(
            ax=ax,
            facecolor='#0f172a',
            edgecolor='#94a3b8',
            linewidth=0.8,
            gid=safe_id
        )
        
        centroid = row.geometry.centroid
        ax.text(
            centroid.x, centroid.y,
            str(city_name).upper(),
            color='white',
            fontsize=8,
            ha='center',
            va='center',
            fontweight='900',
            gid=f"lbl_{safe_id}",
            zorder=10
        )

    # DOSYAYA YAZMA
    with open('id_debug_log.txt', 'w', encoding='utf-8') as f:
        for original, safe in generated_ids:
            f.write(f"{original}|{safe}\n") # Pipe ile ayır, kolay okunsun
            
    # SVG KAYDET
    output_dir = os.path.join('static', 'images')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'turkiye_afet_haritasi.svg')
    
    plt.savefig(
        output_path,
        transparent=True,
        bbox_inches='tight',
        pad_inches=0,
        format='svg'
    )
    plt.close()
    print("✓ HARİTA OLUŞTURULDU!")

if __name__ == "__main__":
    generate_turkey_map()
