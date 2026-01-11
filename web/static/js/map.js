/**
 * Map Page JavaScript - FINAL FIX FOR ISLANDS & MULTIPOLYGONS
 * Solves the "Black Islands" issue (Gökçeada, Bozcaada, etc.)
 * by targeting ALL polygon parts within a city group.
 */

// Global variables
let cityStats = {};
let currentCity = null;
let hoveredCity = null;
let chart = null;

const disasterColors = {
    'Sel': '#4ECDC4',
    'Yangın': '#FF4d4d',
    'Deprem': '#FFa64d',
    'Çığ': '#A569BD'
};

const disasterIcons = { 'Sel': '🌊', 'Yangın': '🔥', 'Deprem': '🏚️', 'Çığ': '🏔️' };

const CITY_MAPPING = {
    'Adana': 'Adana',
    'Adıyaman': 'Adiyaman',
    'Afyonkarahisar': 'Afyon',
    'Ağrı': 'Agri',
    'Aksaray': 'Aksaray',
    'Amasya': 'Amasya',
    'Ankara': 'Ankara',
    'Antalya': 'Antalya',
    'Ardahan': 'Ardahan',
    'Artvin': 'Artvin',
    'Aydın': 'Aydin',
    'Balıkesir': 'Balikesir',
    'Bartın': 'Bartin',
    'Batman': 'Batman',
    'Bayburt': 'Bayburt',
    'Bilecik': 'Bilecik',
    'Bingöl': 'Bingol',
    'Bitlis': 'Bitlis',
    'Bolu': 'Bolu',
    'Burdur': 'Burdur',
    'Bursa': 'Bursa',
    'Çanakkale': 'Canakkale',
    'Çankırı': 'Cankiri',
    'Çorum': 'Corum',
    'Denizli': 'Denizli',
    'Diyarbakır': 'Diyarbakir',
    'Düzce': 'Duzce',
    'Edirne': 'Edirne',
    'Elazığ': 'Elazig',
    'Erzincan': 'Erzincan',
    'Erzurum': 'Erzurum',
    'Eskişehir': 'Eskisehir',
    'Gaziantep': 'Gaziantep',
    'Giresun': 'Giresun',
    'Gümüşhane': 'Gumushane',
    'Hakkari': 'Hakkari',
    'Hatay': 'Hatay',
    'Iğdır': 'Igdir',
    'Isparta': 'Isparta',
    'İstanbul': 'Istanbul',
    'İzmir': 'Izmir',
    'Kahramanmaraş': 'Kahramanmaras',
    'Karabük': 'Karabuk',
    'Karaman': 'Karaman',
    'Kars': 'Kars',
    'Kastamonu': 'Kastamonu',
    'Kayseri': 'Kayseri',
    'Kilis': 'Kilis',
    'Kırıkkale': 'Kirikkale',
    'Kırklareli': 'Kirklareli',
    'Kırşehir': 'Kirsehir',
    'Kocaeli': 'Kocaeli',
    'Konya': 'Konya',
    'Kütahya': 'Kutahya',
    'Malatya': 'Malatya',
    'Manisa': 'Manisa',
    'Mardin': 'Mardin',
    'Mersin': 'Mersin',
    'Muğla': 'Mugla',
    'Muş': 'Mus',
    'Nevşehir': 'Nevsehir',
    'Niğde': 'Nigde',
    'Ordu': 'Ordu',
    'Osmaniye': 'Osmaniye',
    'Rize': 'Rize',
    'Sakarya': 'Sakarya',
    'Samsun': 'Samsun',
    'Şanlıurfa': 'Sanliurfa',
    'Siirt': 'Siirt',
    'Sinop': 'Sinop',
    'Şırnak': 'Sirnak',
    'Sivas': 'Sivas',
    'Tekirdağ': 'Tekirdag',
    'Tokat': 'Tokat',
    'Trabzon': 'Trabzon',
    'Tunceli': 'Tunceli',
    'Uşak': 'Usak',
    'Van': 'Van',
    'Yalova': 'Yalova',
    'Yozgat': 'Yozgat',
    'Zonguldak': 'Zonguldak'
};

function normalizeCityName(name) {
    if (!name) return '';
    return name.replace(/İ/g, 'I').replace(/ı/g, 'i')
        .replace(/Ş/g, 'S').replace(/ş/g, 's')
        .replace(/Ğ/g, 'G').replace(/ğ/g, 'g')
        .replace(/Ü/g, 'U').replace(/ü/g, 'u')
        .replace(/Ö/g, 'O').replace(/ö/g, 'o')
        .replace(/Ç/g, 'C').replace(/ç/g, 'c')
        .replace(/\s+/g, '').trim();
}

document.addEventListener('DOMContentLoaded', async () => {
    await loadCityStats();
    await loadTurkeyMap();
});

async function loadCityStats() {
    try {
        const response = await fetch('/api/city-stats');
        cityStats = await response.json();
    } catch (error) { console.error('Error loading stats:', error); }
}

async function loadTurkeyMap() {
    const wrapper = document.getElementById('turkeyMapWrapper');
    try {
        const response = await fetch('/static/images/turkiye_afet_haritasi.svg?v=' + new Date().getTime());
        if (!response.ok) throw new Error("SVG not found");
        const svgText = await response.text();
        wrapper.innerHTML = svgText;

        const svg = wrapper.querySelector('svg');
        svg.style.width = '100%';
        svg.style.height = 'auto';

        organizeSvgLayers(svg);
        setupMapInteractions(svg);

    } catch (error) {
        console.error("Map load error:", error);
        wrapper.innerHTML = `<p class="error-msg">Harita yüklenemedi. Lütfen sayfayı yenileyin.</p>`;
    }
}

function organizeSvgLayers(svg) {
    let labelsGroup = svg.querySelector('#labels-layer');
    if (!labelsGroup) {
        labelsGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
        labelsGroup.id = "labels-layer";
        svg.appendChild(labelsGroup);
    }
    const allLabels = svg.querySelectorAll('[id^="lbl_"]');
    allLabels.forEach(label => {
        labelsGroup.appendChild(label);
        const textNode = label.tagName === 'text' ? label : label.querySelector('text');
        if (textNode) {
            textNode.style.pointerEvents = 'none';
            textNode.style.fill = '#ffffff';
            textNode.style.fontWeight = '900';
            textNode.style.fontSize = '9px';
            textNode.style.filter = 'drop-shadow(0 1px 2px rgba(0,0,0,0.8))';
        }
    });
}

function findSvgElement(svg, cityName) {
    if (CITY_MAPPING[cityName]) {
        const id = CITY_MAPPING[cityName];
        const el = svg.querySelector(`#${id}`) || svg.querySelector(`g[id="${id}"]`);
        if (el) return el;
    }
    const safeId = normalizeCityName(cityName);
    let el = svg.querySelector(`#${safeId}`) || svg.querySelector(`g[id="${safeId}"]`);
    if (el) return el;

    const allGroups = svg.querySelectorAll('g[id], path[id]');
    for (let i = 0; i < allGroups.length; i++) {
        const id = allGroups[i].id.toLowerCase();
        const search = safeId.toLowerCase();
        if (id === search) return allGroups[i];
    }
    return null;
}

function setupMapInteractions(svg) {
    Object.keys(cityStats).forEach(cityName => {
        try {
            const element = findSvgElement(svg, cityName);
            if (element) {
                setupCityElement(element, cityName, svg);
            }
        } catch (e) {
            console.error(`Error processing city ${cityName}:`, e);
        }
    });
}

function setupCityElement(element, cityName, svg) {
    // === MULTIPOLYGON SUPPORT ===
    // Select all paths inside the group (e.g. Islands)
    let paths = Array.from(element.querySelectorAll('path'));
    if (paths.length === 0) {
        // If element itself is a path
        if (element.tagName === 'path') paths = [element];
    }

    const stats = cityStats[cityName];
    if (!stats || paths.length === 0) return;

    // Find Dominant Disaster
    let dominantDisaster = 'Normal';
    let maxVal = -1;
    for (const [key, val] of Object.entries(stats)) {
        if (val > maxVal) {
            maxVal = val;
            dominantDisaster = key;
        }
    }
    const dominantColor = disasterColors[dominantDisaster] || '#666';

    // Helper to apply style to ALL parts
    const applyStyle = (fill, opacity, stroke, width, glow) => {
        paths.forEach(p => {
            p.style.transition = 'all 0.3s ease';
            p.style.cursor = 'pointer';
            p.style.fill = fill;
            p.style.fillOpacity = opacity;
            p.style.stroke = stroke;
            p.style.strokeWidth = width;
            p.style.filter = glow ? `drop-shadow(0 0 15px ${dominantColor})` : 'none';
        });
    };

    // INITIAL STATE
    applyStyle(dominantColor, '0.4', 'rgba(255,255,255,0.1)', '0.5px', false);

    // Find Label
    const textId = `lbl_${element.id}`;
    const labelsLayer = svg.querySelector('#labels-layer');
    let labelElement = labelsLayer ? labelsLayer.querySelector(`[id="${textId}"]`) : null;

    // Hover
    element.addEventListener('mouseenter', (e) => {
        hoveredCity = element;
        // GLOW
        applyStyle(dominantColor, '0.9', '#fff', '1.5px', true);

        // Label Pop
        if (labelElement) {
            const textNode = labelElement.tagName === 'text' ? labelElement : labelElement.querySelector('text');
            if (textNode) {
                textNode.style.fill = '#fff';
                textNode.style.fontSize = '13px';
                textNode.style.filter = 'drop-shadow(0 0 4px black)';
            }
        }
        handleCityHover(e, cityName, dominantColor, dominantDisaster);
    });

    // Reset
    element.addEventListener('mouseleave', () => {
        hoveredCity = null;
        // RESTING
        applyStyle(dominantColor, '0.4', 'rgba(255,255,255,0.1)', '0.5px', false);

        // Label Reset
        if (labelElement) {
            const textNode = labelElement.tagName === 'text' ? labelElement : labelElement.querySelector('text');
            if (textNode) {
                textNode.style.fill = '#ffffff';
                textNode.style.fontSize = '9px';
                textNode.style.filter = 'drop-shadow(0 1px 2px rgba(0,0,0,0.8))';
            }
        }
        hideTooltip();
    });

    element.addEventListener('click', () => handleCityClick(cityName));
}

function handleCityHover(event, cityName, color, dominant) {
    const tooltip = document.getElementById('mapTooltip');
    tooltip.querySelector('#tooltipCity').textContent = cityName;
    tooltip.querySelector('#tooltipCity').style.color = color;

    let x = event.pageX + 20;
    let y = event.pageY + 10;
    tooltip.style.left = x + 'px';
    tooltip.style.top = y + 'px';
    tooltip.classList.add('show');

    const stats = cityStats[cityName];
    const sorted = Object.entries(stats).sort((a, b) => b[1] - a[1]);
    let html = '';
    sorted.forEach(([d, p]) => {
        const isDom = d === dominant;
        html += `<div style="display:flex;justify-content:space-between;margin:0.2rem 0;font-size:0.9rem;${isDom ? 'font-weight:bold' : ''}">
            <span>${disasterIcons[d]} ${d}:</span><strong style="color:${disasterColors[d]}">${p}%</strong></div>`;
    });
    tooltip.querySelector('#tooltipStats').innerHTML = html;
}

function hideTooltip() { document.getElementById('mapTooltip').classList.remove('show'); }

function handleCityClick(cityName) {
    const modal = document.getElementById('cityModal');
    modal.querySelector('#modalCity').textContent = cityName + ' Afet Analizi';
    createCityChart(cityName);
    displayTopRisk(cityName);
    document.getElementById('googleSearchLink').href = `https://www.google.com/search?q=${encodeURIComponent(cityName + ' doğal afet geçmişi')}`;
    modal.classList.add('active');
    document.getElementById('refreshRecommendations').onclick = () => displayTopRisk(cityName);
}

function closeModal() {
    document.getElementById('cityModal').classList.remove('active');
    if (chart) { chart.destroy(); chart = null; }
}
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });
document.getElementById('cityModal').addEventListener('click', e => { if (e.target.id === 'cityModal') closeModal(); });
const closeBtn = document.querySelector('.modal-close'); if (closeBtn) closeBtn.onclick = closeModal;

function createCityChart(cityName) {
    if (chart) chart.destroy();
    const ctx = document.getElementById('cityChart').getContext('2d');
    const stats = cityStats[cityName];
    const labels = Object.keys(stats), data = Object.values(stats);
    chart = new Chart(ctx, {
        type: 'doughnut',
        data: { labels: labels.map(l => disasterIcons[l] + ' ' + l), datasets: [{ data: data, backgroundColor: labels.map(l => disasterColors[l]), borderColor: '#151934', borderWidth: 2 }] },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom', labels: { color: '#fff' } } }, cutout: '60%' }
    });
}
async function displayTopRisk(cityName) {
    const stats = cityStats[cityName];
    const top = Object.entries(stats).sort((a, b) => b[1] - a[1])[0];
    const color = disasterColors[top[0]];
    const el = document.getElementById('topRisk');
    el.innerHTML = `
        <div style="padding:1.5rem;background:linear-gradient(45deg,${color}33,transparent);border-left:5px solid ${color};border-radius:8px;display:flex;justify-content:space-between;align-items:center">
            <div><div style="font-size:0.8rem;color:#aaa">BASKIN RİSK</div><div style="font-size:1.8rem;color:#fff;font-weight:700">${disasterIcons[top[0]]} ${top[0]}</div></div>
            <div style="font-size:2.5rem;font-weight:800;color:${color}">%${top[1]}</div>
        </div>`;
    try {
        const res = await fetch(`/api/recommendations/${top[0]}`);
        const data = await res.json();
        const recList = document.getElementById('recommendationsList');
        recList.innerHTML = `<div style="background:#ffffff08;padding:15px;border-radius:8px;margin-top:15px">${data.recommendation}</div>`;
    } catch (e) { }
}
