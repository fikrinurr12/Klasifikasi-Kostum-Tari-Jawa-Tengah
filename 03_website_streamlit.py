"""
SCRIPT C: WEBSITE KLASIFIKASI KOSTUM TARI - STREAMLIT
======================================================
Website interaktif untuk klasifikasi kostum tari tradisional Jawa Tengah
menggunakan model CNN yang sudah dilatih.

FITUR:
- Upload & predict gambar kostum tari
- Katalog informasi 5 jenis tarian
- Visualisasi confidence score
- Responsive dan user-friendly

Author: Fasya Maulinada
Tanggal: Januari 2025
"""

# ============================================================================
# INSTALASI & IMPORT
# ============================================================================

# Install di terminal/cmd (bukan di script):
"""
pip install streamlit tensorflow pillow numpy
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
from pathlib import Path
import io

# Set page config (HARUS di paling awal)
st.set_page_config(
    page_title="Klasifikasi Kostum Tari Jawa Tengah",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# KONFIGURASI
# ============================================================================

# Path ke model dan file pendukung
MODEL_PATH = 'final_model.h5'  # Ubah sesuai lokasi model Anda
CLASS_INDICES_PATH = 'class_indices.json'

# Ukuran gambar input model
IMG_SIZE = 224

# Informasi detail setiap tarian
TARI_INFO = {
    'Tari Bedhaya': {
        'deskripsi': '''
        Tari Bedhaya adalah tarian sakral dan tertua yang mencerminkan kerumitan 
        budaya keraton Surakarta dan Yogyakarta. Tarian ini memiliki nilai-nilai 
        edukatif religius, sakral, dan etika kesantunan wanita keraton.
        ''',
        'karakteristik': [
            'Ditarikan oleh 7-9 penari wanita',
            'Gerakan lemah lembut dan khidmat',
            'Kostum didominasi warna hijau atau biru',
            'Menggunakan kain batik motif parang atau kawung',
            'Aksesoris kepala: mahkota atau jamang'
        ],
        'asal': 'Keraton Surakarta & Yogyakarta',
        'image': 'https://via.placeholder.com/400x300?text=Tari+Bedhaya'
    },
    
    'Tari Srimpi': {
        'deskripsi': '''
        Tari Srimpi merupakan tarian putri yang berkarakter lungguh (halus) dan 
        ditarikan secara berkelompok. Sering digunakan untuk menyambut tamu 
        kehormatan di keraton.
        ''',
        'karakteristik': [
            'Ditarikan oleh 4 penari wanita',
            'Gerakan anggun dan simetris',
            'Kostum warna-warni cerah (merah, kuning, hijau, biru)',
            'Masing-masing penari mewakili arah mata angin',
            'Menggunakan sampur (selendang)'
        ],
        'asal': 'Keraton Jawa Tengah',
        'image': 'https://via.placeholder.com/400x300?text=Tari+Srimpi'
    },
    
    'Tari Gambyong': {
        'deskripsi': '''
        Tari Gambyong awalnya tari tunggal putri, tetapi kini sering ditarikan 
        berkelompok untuk pembukaan acara, penyambutan tamu, atau pertunjukan 
        komersial. Berasal dari tarian rakyat (tledhek).
        ''',
        'karakteristik': [
            'Gerakan dinamis dan energik',
            'Kostum didominasi warna cerah (merah, kuning, emas)',
            'Menggunakan sanggul besar dengan bunga melati',
            'Aksesoris berupa kalung, gelang, dan subang',
            'Ekspresi wajah ceria dan sumringah'
        ],
        'asal': 'Surakarta, Jawa Tengah',
        'image': 'https://via.placeholder.com/400x300?text=Tari+Gambyong'
    },
    
    'Tari Golek': {
        'deskripsi': '''
        Tari Golek merupakan tarian klasik yang sangat populer, merepresentasikan 
        remaja putri yang sedang dalam masa pencarian jati diri melalui upaya 
        berhias diri.
        ''',
        'karakteristik': [
            'Tari tunggal atau berpasangan',
            'Gerakan luwes dan gemulai',
            'Kostum didominasi warna pastel (pink, ungu, hijau muda)',
            'Menggunakan kain batik halus',
            'Properti: kipas atau sampur'
        ],
        'asal': 'Surakarta, Jawa Tengah',
        'image': 'https://via.placeholder.com/400x300?text=Tari+Golek'
    },
    
    'Tari Dolalak': {
        'deskripsi': '''
        Tari Dolalak merupakan warisan budaya dari zaman penjajahan Belanda, 
        hasil akulturasi budaya Barat dan Jawa. Tarian ini meniru gerak-gerik 
        serdadu Belanda dengan iringan musik tradisional.
        ''',
        'karakteristik': [
            'Ditarikan oleh kelompok (biasanya wanita)',
            'Gerakan menyerupai marcheren tentara',
            'Kostum unik: perpaduan kebaya dan atribut militer',
            'Menggunakan topi (omprok) khas',
            'Gerakan rampak dan dinamis'
        ],
        'asal': 'Purworejo, Jawa Tengah',
        'image': 'https://via.placeholder.com/400x300?text=Tari+Dolalak'
    }
}

# ============================================================================
# LOAD MODEL & UTILITIES
# ============================================================================

@st.cache_resource
def load_model():
    """Load trained model"""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Pastikan file 'final_model.keras' ada di folder yang sama dengan script ini.")
        return None

@st.cache_data
def load_class_indices():
    """Load class mapping"""
    try:
        with open(CLASS_INDICES_PATH, 'r') as f:
            class_mapping = json.load(f)
        return class_mapping
    except:
        # Default mapping jika file tidak ada
        return {
            'tari_bedhaya': 'Tari Bedhaya',
            'tari_dolalak': 'Tari Dolalak',
            'tari_gambyong': 'Tari Gambyong',
            'tari_golek': 'Tari Golek',
            'tari_srimpi': 'Tari Srimpi'
        }

def preprocess_image(image):
    """Preprocessing gambar untuk prediksi"""
    # Resize ke ukuran yang dibutuhkan model
    img = image.resize((IMG_SIZE, IMG_SIZE))
    
    # Convert ke array
    img_array = np.array(img)
    
    # Pastikan RGB (bukan RGBA)
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    
    # Normalisasi (0-1)
    img_array = img_array / 255.0
    
    # Tambah batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_image(model, image, class_mapping):
    """Prediksi kelas gambar"""
    # Preprocess
    processed_img = preprocess_image(image)
    
    # Predict
    predictions = model.predict(processed_img, verbose=0)
    
    # Get top prediction
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx] * 100
    
    # Map ke nama kelas
    class_keys = list(class_mapping.keys())
    predicted_class_key = class_keys[predicted_class_idx]
    predicted_class_name = class_mapping[predicted_class_key]
    
    # Get all predictions dengan confidence
    all_predictions = []
    for idx, conf in enumerate(predictions[0]):
        class_key = class_keys[idx]
        class_name = class_mapping[class_key]
        all_predictions.append({
            'class': class_name,
            'confidence': conf * 100
        })
    
    # Sort berdasarkan confidence
    all_predictions = sorted(all_predictions, key=lambda x: x['confidence'], reverse=True)
    
    return predicted_class_name, confidence, all_predictions

# ============================================================================
# CUSTOM CSS
# ============================================================================

def load_css():
    """Custom CSS untuk styling"""
    st.markdown("""
    <style>
    /* Main container */
    .main {
        padding: 2rem;
    }
    
    /* Header */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    /* Card styling */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    /* Result container */
    .result-container {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .result-title {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .confidence-score {
        font-size: 3rem;
        font-weight: bold;
    }
    
    /* Progress bar custom */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# PAGE FUNCTIONS
# ============================================================================

def home_page():
    """Halaman utama"""
    # Header
    st.markdown("""
    <div class="header-container">
        <div class="header-title">🎭 Klasifikasi Kostum Tari Tradisional</div>
        <div class="header-subtitle">Jawa Tengah - Powered by AI</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Intro
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 👋 Selamat Datang!")
        st.markdown("""
        Aplikasi ini menggunakan **Deep Learning** dengan arsitektur **MobileNetV2** 
        untuk mengklasifikasikan 5 jenis kostum tari tradisional Jawa Tengah secara otomatis:
        
        - 🎭 **Tari Bedhaya** - Tarian sakral keraton
        - 🎭 **Tari Srimpi** - Tarian empat arah mata angin
        - 🎭 **Tari Gambyong** - Tarian pembukaan yang energik
        - 🎭 **Tari Golek** - Tarian pencarian jati diri
        - 🎭 **Tari Dolalak** - Tarian akulturasi Belanda-Jawa
        
        **Cara menggunakan:**
        1. Pilih menu **"Klasifikasi"** di sidebar
        2. Upload gambar kostum tari
        3. Lihat hasil prediksi AI
        """)
    
    with col2:
        st.image("https://via.placeholder.com/400x500?text=Tari+Jawa+Tengah", 
                use_column_width=True)
    
    # Features
    st.markdown("---")
    st.markdown("### ✨ Fitur Aplikasi")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>🎯 Akurasi Tinggi</h4>
            <p>Model dilatih dengan >1500 gambar dan mencapai akurasi >90%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>⚡ Cepat & Efisien</h4>
            <p>Prediksi instan dalam hitungan detik</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-card">
            <h4>📚 Edukatif</h4>
            <p>Dilengkapi informasi lengkap setiap tarian</p>
        </div>
        """, unsafe_allow_html=True)

def classification_page(model, class_mapping):
    """Halaman klasifikasi"""
    st.title("🎯 Klasifikasi Kostum Tari")
    st.markdown("Upload gambar kostum tari untuk mendapatkan prediksi jenis tariannya.")
    
    # Upload file
    uploaded_file = st.file_uploader(
        "Pilih gambar (JPG, JPEG, PNG)",
        type=['jpg', 'jpeg', 'png'],
        help="Upload gambar yang jelas menampilkan kostum tari"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 📸 Gambar yang Diupload")
            st.image(image, use_column_width=True)
        
        with col2:
            st.markdown("#### 🤖 Hasil Klasifikasi")
            
            with st.spinner('Menganalisis gambar...'):
                # Predict
                predicted_class, confidence, all_predictions = predict_image(
                    model, image, class_mapping
                )
            
            # Display result
            st.markdown(f"""
            <div class="result-container">
                <div class="result-title">{predicted_class}</div>
                <div class="confidence-score">{confidence:.2f}%</div>
                <div>Confidence Score</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence interpretation
            if confidence >= 90:
                st.success("✅ Prediksi sangat yakin!")
            elif confidence >= 70:
                st.info("ℹ️ Prediksi cukup yakin")
            else:
                st.warning("⚠️ Prediksi kurang yakin, coba gambar yang lebih jelas")
        
        # All predictions
        st.markdown("---")
        st.markdown("#### 📊 Semua Prediksi")
        
        for pred in all_predictions:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.progress(pred['confidence'] / 100)
            with col2:
                st.markdown(f"**{pred['confidence']:.1f}%**")
            st.markdown(f"**{pred['class']}**")
            st.markdown("")
        
        # Info tarian yang diprediksi
        if predicted_class in TARI_INFO:
            st.markdown("---")
            st.markdown(f"### 📖 Tentang {predicted_class}")
            
            info = TARI_INFO[predicted_class]
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Deskripsi:**")
                st.markdown(info['deskripsi'])
                
                st.markdown(f"**Karakteristik:**")
                for char in info['karakteristik']:
                    st.markdown(f"- {char}")
                
                st.markdown(f"**Asal:** {info['asal']}")
            
            with col2:
                st.image(info['image'], use_column_width=True)
    
    else:
        # Tampilkan contoh gambar
        st.info("👆 Upload gambar untuk memulai klasifikasi")
        
        st.markdown("### 💡 Contoh Gambar yang Baik:")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("✅ **Pencahayaan Baik**")
            st.image("https://via.placeholder.com/200x200?text=Good+Lighting", 
                    use_column_width=True)
        
        with col2:
            st.markdown("✅ **Kostum Jelas Terlihat**")
            st.image("https://via.placeholder.com/200x200?text=Clear+Costume", 
                    use_column_width=True)
        
        with col3:
            st.markdown("✅ **Fokus pada Penari**")
            st.image("https://via.placeholder.com/200x200?text=Focused", 
                    use_column_width=True)

def catalog_page():
    """Halaman katalog tarian"""
    st.title("📚 Katalog Tari Tradisional Jawa Tengah")
    st.markdown("Pelajari lebih dalam tentang 5 tarian tradisional yang dapat diklasifikasi sistem ini.")
    
    # Tabs untuk setiap tarian
    tabs = st.tabs([name for name in TARI_INFO.keys()])
    
    for idx, (tari_name, info) in enumerate(TARI_INFO.items()):
        with tabs[idx]:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"## {tari_name}")
                
                st.markdown("### 📝 Deskripsi")
                st.markdown(info['deskripsi'])
                
                st.markdown("### 🎨 Karakteristik")
                for char in info['karakteristik']:
                    st.markdown(f"- {char}")
                
                st.markdown(f"### 📍 Asal")
                st.markdown(info['asal'])
            
            with col2:
                st.image(info['image'], use_column_width=True, 
                        caption=f"Ilustrasi {tari_name}")
                
                # Fun fact atau info tambahan bisa ditambahkan di sini
                with st.expander("ℹ️ Tahukah Anda?"):
                    st.markdown(f"""
                    {tari_name} memiliki sejarah panjang dalam budaya Jawa Tengah 
                    dan masih aktif dipentaskan hingga saat ini dalam berbagai 
                    acara budaya dan pariwisata.
                    """)

def about_page():
    """Halaman tentang aplikasi"""
    st.title("ℹ️ Tentang Aplikasi")
    
    st.markdown("""
    ### 🎓 Latar Belakang Penelitian
    
    Aplikasi ini dikembangkan sebagai bagian dari penelitian skripsi untuk 
    **melestarikan budaya tari tradisional Jawa Tengah** melalui teknologi 
    **Artificial Intelligence (AI)**.
    
    ### 🔬 Metodologi
    
    #### 1. Dataset
    - Total: **1,500 gambar** (300 per kelas)
    - Sumber: Scraping dari internet dan dokumentasi sanggar
    - Pembagian: 70% Training | 15% Validation | 15% Testing
    
    #### 2. Model Deep Learning
    - **Arsitektur:** MobileNetV2 (Transfer Learning)
    - **Framework:** TensorFlow/Keras
    - **Input:** 224x224 RGB images
    - **Output:** 5 classes (Softmax activation)
    
    #### 3. Training Process
    - **Fase 1:** Feature Extraction (30 epochs)
    - **Fase 2:** Fine-Tuning (50 epochs)
    - **Optimizer:** Adam
    - **Loss Function:** Categorical Crossentropy
    
    #### 4. Hasil Evaluasi
    """)
    
    # Tampilkan metrik (ganti dengan hasil aktual)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", "92.5%", "+2.5%")
    with col2:
        st.metric("Precision", "91.8%", "+1.8%")
    with col3:
        st.metric("Recall", "92.1%", "+2.1%")
    with col4:
        st.metric("F1-Score", "91.9%", "+1.9%")
    
    st.markdown("""
    ### 👨‍💻 Pengembang
    
    **Fasya Maulinada**  
    NIM: 202251155  
    Program Studi Teknik Informatika  
    Universitas Muria Kudus
    
    **Dosen Pembimbing:**
    - Dr. Ahmad Abdul Chamid, S.Kom, M.Kom
    - Dr. Ahmad Jazuli, S.Kom, M.Kom
    
    ### 🛠️ Teknologi yang Digunakan
    
    - **Backend:** Python, TensorFlow, Keras
    - **Frontend:** Streamlit
    - **Deployment:** Streamlit Cloud
    - **Storage:** Google Drive
    
    ### 📞 Kontak
    
    Untuk pertanyaan atau saran, silakan hubungi:
    - Email: fasya.maulinada@example.com (ganti dengan email asli)
    - GitHub: github.com/fasyamaulinada (ganti dengan GitHub asli)
    
    ### 🙏 Acknowledgments
    
    Terima kasih kepada:
    - Universitas Muria Kudus
    - Dinas Kebudayaan Jawa Tengah
    - Sanggar-sanggar tari yang mendukung penelitian
    - Semua pihak yang telah membantu
    """)

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application"""
    
    # Load CSS
    load_css()
    
    # Load model
    model = load_model()
    class_mapping = load_class_indices()
    
    # Sidebar
    st.sidebar.title("🎭 Navigation")
    
    # Menu selection
    menu = st.sidebar.radio(
        "Pilih Menu:",
        ["🏠 Beranda", "🎯 Klasifikasi", "📚 Katalog", "ℹ️ Tentang"],
        index=0
    )
    
    # Info di sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Informasi Sistem")
    
    if model is not None:
        st.sidebar.success("✅ Model: Loaded")
        st.sidebar.info(f"📦 Classes: {len(class_mapping)}")
    else:
        st.sidebar.error("❌ Model: Not Loaded")
        st.sidebar.warning("Pastikan file model tersedia")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style='text-align: center; font-size: 0.8rem; color: gray;'>
    © 2025 Fasya Maulinada<br>
    Universitas Muria Kudus
    </div>
    """, unsafe_allow_html=True)
    
    # Route to pages
    if menu == "🏠 Beranda":
        home_page()
    
    elif menu == "🎯 Klasifikasi":
        if model is not None:
            classification_page(model, class_mapping)
        else:
            st.error("❌ Model tidak dapat dimuat. Pastikan file model tersedia.")
    
    elif menu == "📚 Katalog":
        catalog_page()
    
    elif menu == "ℹ️ Tentang":
        about_page()

if __name__ == "__main__":
    main()


# ============================================================================
# CARA MENJALANKAN
# ============================================================================
"""
LANGKAH DEPLOYMENT:

1. PERSIAPAN FILE:
   Pastikan file berikut ada di folder yang sama:
   - 03_website_streamlit.py (script ini)
   - final_model.keras (model hasil training)
   - class_indices.json (mapping kelas)

2. INSTALL DEPENDENCIES:
   pip install streamlit tensorflow pillow numpy

3. JALANKAN LOKAL:
   streamlit run 03_website_streamlit.py

4. DEPLOYMENT KE STREAMLIT CLOUD:
   
   a. Buat file requirements.txt:
      ```
      streamlit==1.31.0
      tensorflow==2.15.0
      pillow==10.2.0
      numpy==1.26.3
      ```
   
   b. Upload file ke GitHub:
      - 03_website_streamlit.py
      - final_model.keras
      - class_indices.json
      - requirements.txt
   
   c. Deploy di Streamlit Cloud:
      - Buka https://streamlit.io/cloud
      - Connect dengan GitHub
      - Pilih repository
      - Deploy!
   
   d. CATATAN: Model keras (.keras) cukup besar (~14MB)
      Jika GitHub menolak, alternatif:
      - Upload model ke Google Drive
      - Modify code untuk download model dari Drive
      - Atau gunakan model TFLite yang lebih kecil

5. HOSTING MODEL DI GOOGLE DRIVE:
   
   Jika model terlalu besar untuk GitHub:
   
   ```python
   import gdown
   
   # Di awal script, tambahkan:
   MODEL_URL = 'https://drive.google.com/uc?id=YOUR_FILE_ID'
   MODEL_PATH = 'final_model.keras'
   
   @st.cache_resource
   def load_model():
       if not os.path.exists(MODEL_PATH):
           with st.spinner('Downloading model...'):
               gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
       return tf.keras.models.load_model(MODEL_PATH)
   ```

TROUBLESHOOTING:

Q: Error "ModuleNotFoundError: No module named 'tensorflow'"
A: Install: pip install tensorflow

Q: Model terlalu besar untuk upload
A: Gunakan model TFLite atau host di Google Drive

Q: Streamlit Cloud out of memory
A: Optimize model atau upgrade plan

Q: Gambar tidak muncul
A: Ganti placeholder URL dengan gambar asli dari dokumentasi

TIPS OPTIMIZATION:

1. Gunakan st.cache untuk fungsi berat
2. Compress gambar sebelum upload
3. Lazy load image di katalog
4. Monitor memory usage di Streamlit Cloud

NEXT STEPS:

1. Ganti placeholder images dengan gambar asli
2. Tambah fitur download hasil klasifikasi
3. Tambah history klasifikasi
4. Implementasi feedback mechanism
5. Add analytics untuk tracking usage
"""
