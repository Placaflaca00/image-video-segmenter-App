import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
import os
from scipy import ndimage
from skimage import filters, measure, morphology, color
from skimage.segmentation import watershed, felzenszwalb, slic, quickshift, find_boundaries
from skimage.feature import peak_local_max
import tempfile
import time
import pandas as pd

# Configuración de la página
st.set_page_config(
    page_title="Segmentación de Imágenes - PoC",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personalizado
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    .main-title {
        text-align: center;
        color: #ffffff;
        font-size: 3.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        background: linear-gradient(45deg, #00ff88, #00ccff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 20px;
    }
    
    .subtitle {
        text-align: center;
        color: #e0e0e0;
        font-size: 1.3rem;
        margin-bottom: 2rem;
    }
    
    .image-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 20px;
        border: 2px solid rgba(0, 255, 136, 0.3);
        box-shadow: 0 10px 40px rgba(0, 255, 136, 0.1);
    }
    
    .control-panel {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 20px;
        border: 2px solid rgba(0, 204, 255, 0.3);
        box-shadow: 0 10px 40px rgba(0, 204, 255, 0.1);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px;
        border-radius: 10px;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #00ff88;
    }
    
    .metric-label {
        color: #e0e0e0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Inicializar estado
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}
if 'current_technique' not in st.session_state:
    st.session_state.current_technique = None
if 'processed_video_path' not in st.session_state:
    st.session_state.processed_video_path = None
if 'is_video_mode' not in st.session_state:
    st.session_state.is_video_mode = False
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = None

# Funciones de segmentación
def detect_points(image):
    """Detección de puntos usando ORB"""
    start_time = time.time()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    
    orb = cv2.ORB_create(
        nfeatures=1000,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31
    )
    
    kp = orb.detect(gray, None)
    kp, des = orb.compute(gray, kp)
    
    result = image.copy()
    for keypoint in kp:
        x, y = keypoint.pt
        cv2.circle(result, (int(x), int(y)), 3, (0, 255, 0), -1)
    
    exec_time = (time.time() - start_time) * 1000
    
    metrics = {
        "Puntos detectados": len(kp),
        "Tiempo (ms)": f"{exec_time:.2f}",
        "Densidad": f"{(len(kp) / (image.shape[0] * image.shape[1]) * 10000):.2f} pts/100px²"
    }
    
    return result, metrics

def detect_lines(image):
    """Detección de líneas usando Hough"""
    start_time = time.time()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    
    # Mejora de contraste
    gray = cv2.equalizeHist(gray)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    
    edges = cv2.Canny(gray, 50, 150, apertureSize=3, L2gradient=True)
    
    # Dilatación
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                            minLineLength=30, maxLineGap=15)
    
    result = image.copy()
    line_count = 0
    
    if lines is not None:
        filtered_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if length > 40:
                filtered_lines.append(line)
        
        for line in filtered_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        line_count = len(filtered_lines)
    
    exec_time = (time.time() - start_time) * 1000
    
    metrics = {
        "Líneas detectadas": line_count,
        "Tiempo (ms)": f"{exec_time:.2f}",
        "Long. promedio": f"{np.mean([np.sqrt((l[0][2]-l[0][0])**2 + (l[0][3]-l[0][1])**2) for l in lines]):.1f}px" if lines is not None else "0"
    }
    
    return result, metrics

def detect_edges(image):
    """Detección de bordes usando múltiples operadores"""
    start_time = time.time()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    
    # Canny (principal)
    edges = cv2.Canny(gray, 50, 150)
    
    # Overlay edges on original image
    result = image.copy()
    result[edges > 0] = [0, 255, 0]
    
    exec_time = (time.time() - start_time) * 1000
    edge_pixels = np.sum(edges > 0)
    
    metrics = {
        "Píxeles de borde": edge_pixels,
        "Tiempo (ms)": f"{exec_time:.2f}",
        "Densidad": f"{(edge_pixels / edges.size * 100):.1f}%"
    }
    
    return result, metrics

def combined_detection(image):
    """Detección combinada: bordes + líneas + puntos"""
    start_time = time.time()
    result = image.copy()
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    
    # 1. Bordes en azul
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    result[edges > 0] = [0, 100, 255]
    
    # 2. Líneas en verde
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                            minLineLength=50, maxLineGap=10)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # 3. Puntos en rojo
    orb = cv2.ORB_create(nfeatures=200)
    kp = orb.detect(gray, None)
    for keypoint in kp:
        x, y = keypoint.pt
        cv2.circle(result, (int(x), int(y)), 4, (255, 0, 0), -1)
    
    exec_time = (time.time() - start_time) * 1000
    
    metrics = {
        "Total elementos": len(kp) + (len(lines) if lines is not None else 0),
        "Tiempo (ms)": f"{exec_time:.2f}",
        "Complejidad": "Alta"
    }
    
    return result, metrics

def otsu_threshold(image):
    """Umbralización OTSU - Imagen binaria"""
    start_time = time.time()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    
    # Aplicar OTSU
    threshold_value = filters.threshold_otsu(gray)
    binary = gray > threshold_value
    
    # Convertir a imagen RGB para visualización (blanco y negro)
    result = np.zeros_like(image)
    result[binary] = [255, 255, 255]  # Blanco donde es True
    result[~binary] = [0, 0, 0]  # Negro donde es False
    
    exec_time = (time.time() - start_time) * 1000
    
    metrics = {
        "Umbral": f"{threshold_value:.0f}",
        "Tiempo (ms)": f"{exec_time:.2f}",
        "Ratio B/N": f"{np.sum(binary) / binary.size:.2%}"
    }
    
    return result, metrics

def adaptive_threshold(image):
    """Umbralización adaptativa - Imagen binaria"""
    start_time = time.time()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    
    # Aplicar umbralización adaptativa
    threshold_adaptive = filters.threshold_local(gray, block_size=35, offset=10)
    binary = gray > threshold_adaptive
    
    # Convertir a imagen RGB para visualización
    result = np.zeros_like(image)
    result[binary] = [255, 255, 255]
    result[~binary] = [0, 0, 0]
    
    exec_time = (time.time() - start_time) * 1000
    
    metrics = {
        "Block size": 35,
        "Tiempo (ms)": f"{exec_time:.2f}",
        "Píxeles blancos": f"{np.sum(binary):,}"
    }
    
    return result, metrics

def region_growing(image):
    """Segmentación por crecimiento de regiones usando watershed"""
    start_time = time.time()
    
    # Asegurar que trabajamos con una copia
    image = np.array(image, dtype=np.uint8)
    
    # Convertir a escala de grises si es necesario
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Aplicar filtro gaussiano para reducir ruido
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Threshold binario usando OTSU
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Operaciones morfológicas para limpiar la imagen
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Área de fondo seguro
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Área de primer plano seguro usando transformada de distancia
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.3*dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # Región desconocida
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Etiquetado de componentes conectados
    _, markers = cv2.connectedComponents(sure_fg)
    
    # Añadir 1 a todas las etiquetas para que el fondo sea 1, no 0
    markers = markers + 1
    
    # Marcar la región desconocida con 0
    markers[unknown == 255] = 0
    
    # Preparar imagen para watershed (necesita ser BGR)
    if len(image.shape) == 2:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Aplicar watershed
    markers = cv2.watershed(image_bgr, markers.copy())
    
    # Crear imagen resultado
    result = np.zeros_like(image, dtype=np.uint8)
    
    # Obtener regiones únicas (excluyendo fondo y bordes)
    unique_markers = np.unique(markers)
    valid_markers = [m for m in unique_markers if m > 1]
    
    # Generar colores aleatorios para cada región
    np.random.seed(42)  # Para colores consistentes
    if len(image.shape) == 3:
        colors = np.random.randint(50, 255, size=(len(valid_markers), 3), dtype=np.uint8)
    else:
        colors = np.random.randint(50, 255, size=len(valid_markers), dtype=np.uint8)
    
    # Aplicar colores a las regiones
    for idx, marker in enumerate(valid_markers):
        mask = markers == marker
        if len(image.shape) == 3:
            result[mask] = colors[idx]
        else:
            result[mask] = colors[idx]
    
    # Dibujar bordes en blanco
    if len(image.shape) == 3:
        result[markers == -1] = [255, 255, 255]
    else:
        result[markers == -1] = 255
    
    # Si la imagen original era RGB, asegurar que el resultado también lo sea
    if len(image.shape) == 3 and len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    
    exec_time = (time.time() - start_time) * 1000
    
    # Calcular métricas
    num_regions = len(valid_markers)
    total_pixels = np.sum(markers > 1)
    avg_area = total_pixels / max(num_regions, 1)
    
    metrics = {
        "Regiones": num_regions,
        "Tiempo (ms)": f"{exec_time:.2f}",
        "Área promedio": f"{avg_area:.0f}px"
    }
    
    return result, metrics

def split_merge(image):
    """Segmentación por división y fusión usando superpixels"""
    start_time = time.time()
    
    segments = slic(image, n_segments=100, compactness=10, sigma=1)
    
    result = image.copy()
    for segment_val in np.unique(segments):
        mask = segments == segment_val
        result[mask] = np.mean(image[mask], axis=0)
    
    boundaries = find_boundaries(segments, mode='thick')
    result[boundaries] = [0, 255, 0]
    
    exec_time = (time.time() - start_time) * 1000
    
    metrics = {
        "Segmentos": len(np.unique(segments)),
        "Tiempo (ms)": f"{exec_time:.2f}",
        "Compactness": 10
    }
    
    return result, metrics

def process_video_motion(video_path, progress_callback=None):
    """Detección de movimiento en video - Dominio Espacial"""
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None
        
        # Obtener propiedades del video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 30
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = min(50, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        
        if width <= 0 or height <= 0:
            cap.release()
            return None
        
        # Configurar detector de movimiento
        backSub = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=False
        )
        
        # Intentar diferentes formatos de salida
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.avi').name
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            cap.release()
            return None
        
        frame_count = 0
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detección de movimiento
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask = backSub.apply(gray)
            
            # Limpiar ruido con operaciones morfológicas
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Threshold binario
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            
            # Crear visualización
            result = frame.copy()
            # Resaltar áreas con movimiento en rojo
            result[mask > 0] = [0, 0, 255]
            
            # Mezclar con frame original para mejor visualización
            result = cv2.addWeighted(frame, 0.6, result, 0.4, 0)
            
            out.write(result)
            
            if progress_callback:
                progress_callback((i + 1) / total_frames)
        
        cap.release()
        out.release()
        
        # Verificar que el archivo se creó correctamente
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0 and frame_count >= 10:
            return output_path
        
        return None
        
    except Exception as e:
        print(f"Error en process_video_motion: {e}")
        return None

def process_video_frequency(video_path, progress_callback=None):
    """Análisis de frecuencia temporal en video - Dominio de Frecuencia"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    total_frames = min(50, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    
    # Leer todos los frames primero
    frames_original = []
    frames_gray = []
    
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames_original.append(frame)  # Guardar frame original
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames_gray.append(gray)
        
        if progress_callback:
            progress_callback((i + 1) / (total_frames * 2))
    
    cap.release()  # Cerrar el video original
    
    if len(frames_gray) < 10:
        return None
    
    height, width = frames_gray[0].shape
    
    # Análisis FFT temporal
    frames_array = np.array(frames_gray)
    
    # FFT en la dimensión temporal
    fft_temporal = np.fft.fft(frames_array, axis=0)
    fft_magnitude = np.abs(fft_temporal)
    
    # Detectar frecuencias de movimiento
    motion_freq = np.mean(fft_magnitude[1:len(frames_gray)//2], axis=0)
    
    # Normalizar
    if motion_freq.max() > motion_freq.min():
        motion_freq = (motion_freq - motion_freq.min()) / (motion_freq.max() - motion_freq.min())
    else:
        motion_freq = np.zeros_like(motion_freq)
    
    # Crear threshold para movimiento
    threshold = np.percentile(motion_freq, 70)
    motion_mask = (motion_freq > threshold).astype(np.uint8) * 255
    
    # Preparar video de salida
    fps = 25  # FPS por defecto
    if len(frames_original) > 0:
        height, width = frames_original[0].shape[:2]
    
    # Crear archivo de salida
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.avi').name
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Codec más compatible
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        return None
    
    # Escribir frames procesados
    for i, frame_orig in enumerate(frames_original):
        # Crear overlay verde para movimiento
        overlay = np.zeros_like(frame_orig)
        overlay[:, :, 1] = motion_mask  # Canal verde
        
        # Mezclar con imagen original
        result = cv2.addWeighted(frame_orig, 0.7, overlay, 0.3, 0)
        
        out.write(result)
        
        if progress_callback:
            progress_callback(0.5 + (i + 1) / (len(frames_original) * 2))
    
    out.release()
    
    # Verificar que el archivo se creó
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return output_path
    
    return None

def compare_all_techniques(image):
    """Compara todas las técnicas"""
    techniques = [
        ("Puntos", detect_points),
        ("Líneas", detect_lines),
        ("Bordes", detect_edges),
        ("Combinada", combined_detection),
        ("OTSU", otsu_threshold),
        ("Adaptativa", adaptive_threshold),
        ("Regiones", region_growing),
        ("Superpixels", split_merge)
    ]
    
    results = []
    for name, func in techniques:
        _, metrics = func(image)
        time_ms = float(metrics.get("Tiempo (ms)", "0"))
        results.append({
            "Técnica": name,
            "Tiempo (ms)": time_ms,
            "Eficiencia": "⚡ Rápida" if time_ms < 20 else "🔄 Media" if time_ms < 50 else "⏱️ Lenta"
        })
    
    return pd.DataFrame(results)

# Interfaz principal
def main():
    st.markdown('<h1 class="main-title">🎯 Segmentación de Imágenes</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Procesamiento Digital de Imágenes - Workshop PoC</p>', unsafe_allow_html=True)
    
    col_image, col_control = st.columns([1.2, 0.8], gap="medium")
    
    with col_image:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.markdown("### 📷 Visualización")
        
        image_placeholder = st.empty()
        metrics_placeholder = st.empty()
        
        # Mostrar video procesado o imagen según el modo
        if st.session_state.is_video_mode and st.session_state.processed_video_path:
            # Modo video: mostrar solo el video
            image_placeholder.video(st.session_state.processed_video_path)
        elif st.session_state.current_image is not None and not st.session_state.is_video_mode:
            # Modo imagen: mostrar la imagen
            image_placeholder.image(st.session_state.current_image, use_container_width=True)
            
            if st.session_state.metrics:
                with metrics_placeholder.container():
                    st.markdown("#### 📊 Métricas")
                    cols = st.columns(len(st.session_state.metrics))
                    for i, (label, value) in enumerate(st.session_state.metrics.items()):
                        with cols[i]:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">{value}</div>
                                <div class="metric-label">{label}</div>
                            </div>
                            """, unsafe_allow_html=True)
        else:
            image_placeholder.info("👈 Carga una imagen o video para comenzar")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_control:
        st.markdown('<div class="control-panel">', unsafe_allow_html=True)
        st.markdown("### 🛠️ Panel de Control")
        
        tab_image, tab_video = st.tabs(["📸 Imagen", "🎬 Video"])
        
        with tab_image:
            uploaded_file = st.file_uploader("Cargar imagen", type=['png', 'jpg', 'jpeg', 'bmp'])
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                image = np.array(image)
                
                if st.session_state.original_image is None or not np.array_equal(st.session_state.original_image, image):
                    st.session_state.original_image = image.copy()
                    st.session_state.current_image = image
                    st.session_state.metrics = {}
                    st.session_state.is_video_mode = False
                    st.session_state.processed_video_path = None
                    st.rerun()
                
                st.success("✅ Imagen lista")
                
                st.markdown("#### 🎨 Aplicar Técnica")
                st.warning("⚠️ Presiona 'Restaurar' antes de aplicar una nueva técnica")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📍 Puntos", key="pts", use_container_width=True):
                        if st.session_state.current_image is not None:
                            result, metrics = detect_points(st.session_state.current_image)
                            st.session_state.current_image = result
                            st.session_state.metrics = metrics
                            st.session_state.is_video_mode = False
                            st.rerun()
                    
                    if st.button("📏 Líneas", key="lns", use_container_width=True):
                        if st.session_state.current_image is not None:
                            result, metrics = detect_lines(st.session_state.current_image)
                            st.session_state.current_image = result
                            st.session_state.metrics = metrics
                            st.session_state.is_video_mode = False
                            st.rerun()
                    
                    if st.button("🔲 Bordes", key="edg", use_container_width=True):
                        if st.session_state.current_image is not None:
                            result, metrics = detect_edges(st.session_state.current_image)
                            st.session_state.current_image = result
                            st.session_state.metrics = metrics
                            st.session_state.is_video_mode = False
                            st.rerun()
                    
                    if st.button("🌈 Combinada", key="cmb", use_container_width=True):
                        if st.session_state.current_image is not None:
                            result, metrics = combined_detection(st.session_state.current_image)
                            st.session_state.current_image = result
                            st.session_state.metrics = metrics
                            st.session_state.is_video_mode = False
                            st.rerun()
                
                with col2:
                    if st.button("⚪ OTSU", key="ots", use_container_width=True):
                        if st.session_state.current_image is not None:
                            result, metrics = otsu_threshold(st.session_state.current_image)
                            st.session_state.current_image = result
                            st.session_state.metrics = metrics
                            st.session_state.is_video_mode = False
                            st.rerun()
                    
                    if st.button("🔧 Adaptativa", key="adp", use_container_width=True):
                        if st.session_state.current_image is not None:
                            result, metrics = adaptive_threshold(st.session_state.current_image)
                            st.session_state.current_image = result
                            st.session_state.metrics = metrics
                            st.session_state.is_video_mode = False
                            st.rerun()
                    
                    if st.button("🌱 Regiones", key="rgn", use_container_width=True):
                        if st.session_state.current_image is not None:
                            result, metrics = region_growing(st.session_state.current_image)
                            st.session_state.current_image = result
                            st.session_state.metrics = metrics
                            st.session_state.is_video_mode = False
                            st.rerun()
                    
                    if st.button("✂️ Superpixels", key="spl", use_container_width=True):
                        if st.session_state.current_image is not None:
                            result, metrics = split_merge(st.session_state.current_image)
                            st.session_state.current_image = result
                            st.session_state.metrics = metrics
                            st.session_state.is_video_mode = False
                            st.rerun()
                
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Restaurar", use_container_width=True):
                        st.session_state.current_image = st.session_state.original_image.copy()
                        st.session_state.metrics = {}
                        st.rerun()
                
                with col2:
                    if st.button("📊 Comparar", use_container_width=True):
                        with st.spinner("Analizando..."):
                            st.session_state.comparison_results = compare_all_techniques(st.session_state.original_image)
                        st.rerun()
                
                if st.session_state.comparison_results is not None:
                    st.markdown("#### 📈 Comparación Real")
                    st.dataframe(st.session_state.comparison_results, use_container_width=True, hide_index=True)
        
        with tab_video:
            uploaded_video = st.file_uploader("Cargar video", type=['mp4', 'avi', 'mov'])
            
            if uploaded_video is not None:
                # Guardar el video en un archivo temporal
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_video.read())
                tfile.close()
                
                st.success("✅ Video cargado")
                
                # Mostrar el video original
                st.markdown("#### 🎥 Video Original")
                st.video(tfile.name)
                
                st.markdown("---")
                st.markdown("#### 🎬 Procesar Video")
                st.info("💡 El procesamiento analizará los primeros 50 frames del video")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🏃 Mov. Espacial", key="mot", use_container_width=True):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        def update_progress(progress):
                            progress_bar.progress(progress)
                            status_text.text(f"Procesando: {int(progress*100)}%")
                        
                        try:
                            output = process_video_motion(tfile.name, update_progress)
                            
                            if output and os.path.exists(output):
                                st.session_state.processed_video_path = output
                                st.session_state.is_video_mode = True
                                st.session_state.current_image = None
                                status_text.success("✅ Video procesado - Movimiento detectado en rojo")
                                time.sleep(1)
                                st.rerun()
                            else:
                                status_text.error("❌ Error al procesar el video")
                        except Exception as e:
                            status_text.error(f"❌ Error: {str(e)}")
                
                with col2:
                    if st.button("📊 Mov. Frecuencia", key="freq", use_container_width=True):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        def update_progress(progress):
                            progress_bar.progress(progress)
                            status_text.text(f"Analizando FFT: {int(progress*100)}%")
                        
                        try:
                            output = process_video_frequency(tfile.name, update_progress)
                            
                            if output and os.path.exists(output):
                                st.session_state.processed_video_path = output
                                st.session_state.is_video_mode = True
                                st.session_state.current_image = None
                                status_text.success("✅ Análisis completado - Frecuencias en verde")
                                time.sleep(1)
                                st.rerun()
                            else:
                                status_text.error("❌ Error al procesar el video")
                        except Exception as e:
                            status_text.error(f"❌ Error: {str(e)}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Limpiar Video", use_container_width=True):
                        # Limpiar archivos temporales si existen
                        if st.session_state.processed_video_path and os.path.exists(st.session_state.processed_video_path):
                            try:
                                os.remove(st.session_state.processed_video_path)
                            except:
                                pass
                        
                        st.session_state.processed_video_path = None
                        st.session_state.is_video_mode = False
                        if st.session_state.original_image is not None:
                            st.session_state.current_image = st.session_state.original_image.copy()
                        st.rerun()
                
                with col2:
                    # Botón para descargar video procesado
                    if st.session_state.processed_video_path and os.path.exists(st.session_state.processed_video_path):
                        with open(st.session_state.processed_video_path, 'rb') as f:
                            st.download_button(
                                label="💾 Descargar Video",
                                data=f.read(),
                                file_name="video_procesado.mp4",
                                mime="video/mp4",
                                use_container_width=True
                            )
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()