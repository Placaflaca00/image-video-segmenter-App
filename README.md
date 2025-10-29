# 🎯 Segmentación de Imágenes y Videos

Aplicación de segmentación inteligente con 8+ algoritmos de Computer Vision para procesamiento de imágenes y análisis de movimiento en videos. Desarrollada con Streamlit, OpenCV y scikit-image.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)

## 📋 Características

- ✅ **8 Técnicas de Segmentación de Imágenes**
- ✅ **Análisis de Movimiento en Videos** (Espacial y Frecuencial)
- ✅ **Métricas en Tiempo Real**
- ✅ **Comparación de Algoritmos**
- ✅ **Interfaz Interactiva y Moderna**
- ✅ **Exportación de Resultados**

---

## 🚀 Instalación y Configuración

### 1. Clonar el Repositorio

```bash
git clone https://github.com/tu-usuario/tu-repo.git
cd tu-repo
```

### 2. Crear Entorno Virtual

#### En Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### En Linux/Mac:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 4. Ejecutar la Aplicación

```bash
streamlit run app.py
```

### 5. Acceder a la Aplicación

La aplicación se abrirá automáticamente en tu navegador en:

```
🌐 Local URL: http://localhost:8501
🔗 Network URL: http://192.168.x.x:8501
```

Si no se abre automáticamente, copia y pega la URL en tu navegador.

---

## 📦 Dependencias Principales

```txt
streamlit
opencv-python
numpy
matplotlib
Pillow
scipy
scikit-image
pandas
```

---

## 🛠️ Métodos y Técnicas Implementadas

### 📸 **Segmentación de Imágenes**

#### 1. **Detección de Puntos (ORB)** 🔴
```python
detect_points(image)
```
- **Algoritmo**: ORB (Oriented FAST and Rotated BRIEF)
- **Función**: Detecta puntos de interés/características clave en la imagen
- **Parámetros**: 1000 features, 8 niveles de pirámide
- **Uso**: Reconocimiento de objetos, matching de imágenes
- **Métricas**: 
  - Puntos detectados
  - Densidad de puntos por área
  - Tiempo de ejecución

#### 2. **Detección de Líneas (Hough Transform)** 📏
```python
detect_lines(image)
```
- **Algoritmo**: Transformada de Hough Probabilística
- **Función**: Detecta líneas rectas en la imagen
- **Preprocesamiento**: 
  - Ecualización de histograma
  - Filtro bilateral
  - Detección de bordes con Canny
- **Parámetros**: 
  - Threshold: 50
  - Longitud mínima: 30px
  - Gap máximo: 15px
- **Uso**: Detección de estructuras, análisis arquitectónico
- **Métricas**:
  - Número de líneas detectadas
  - Longitud promedio
  - Tiempo de ejecución

#### 3. **Detección de Bordes (Canny)** 🔲
```python
detect_edges(image)
```
- **Algoritmo**: Detector de bordes Canny
- **Función**: Detecta contornos y bordes en la imagen
- **Parámetros**: 
  - Threshold bajo: 50
  - Threshold alto: 150
- **Uso**: Segmentación de objetos, análisis de formas
- **Métricas**:
  - Píxeles de borde detectados
  - Densidad de bordes
  - Tiempo de ejecución

#### 4. **Detección Combinada** 🌈
```python
combined_detection(image)
```
- **Algoritmo**: Combinación de Canny + Hough + ORB
- **Función**: Aplica múltiples técnicas simultáneamente
- **Visualización**:
  - Bordes en **azul**
  - Líneas en **verde**
  - Puntos en **rojo**
- **Uso**: Análisis completo de la escena
- **Métricas**:
  - Total de elementos detectados
  - Complejidad computacional
  - Tiempo de ejecución

#### 5. **Umbralización OTSU** ⚪
```python
otsu_threshold(image)
```
- **Algoritmo**: Método de Otsu para binarización automática
- **Función**: Separa objetos del fondo mediante umbral óptimo
- **Salida**: Imagen binaria (blanco/negro)
- **Uso**: Separación de objetos, análisis de documentos
- **Métricas**:
  - Valor del umbral calculado
  - Ratio blanco/negro
  - Tiempo de ejecución

#### 6. **Umbralización Adaptativa** 🔧
```python
adaptive_threshold(image)
```
- **Algoritmo**: Umbralización local adaptativa
- **Función**: Calcula umbrales diferentes para cada región
- **Parámetros**:
  - Block size: 35
  - Offset: 10
- **Ventaja**: Mejor rendimiento en iluminación no uniforme
- **Uso**: Escaneo de documentos, detección de texto
- **Métricas**:
  - Tamaño de bloque usado
  - Píxeles blancos totales
  - Tiempo de ejecución

#### 7. **Segmentación por Regiones (Watershed)** 🌱
```python
region_growing(image)
```
- **Algoritmo**: Watershed (Cuencas Hidrográficas)
- **Función**: Segmenta la imagen en regiones homogéneas
- **Proceso**:
  1. Filtrado Gaussiano
  2. Umbralización OTSU
  3. Operaciones morfológicas
  4. Transformada de distancia
  5. Algoritmo Watershed
- **Salida**: Regiones coloreadas con bordes blancos
- **Uso**: Conteo de objetos, análisis celular
- **Métricas**:
  - Número de regiones detectadas
  - Área promedio por región
  - Tiempo de ejecución

#### 8. **Segmentación por Superpixels (SLIC)** ✂️
```python
split_merge(image)
```
- **Algoritmo**: SLIC (Simple Linear Iterative Clustering)
- **Función**: Agrupa píxeles en superpixels coherentes
- **Parámetros**:
  - 100 segmentos
  - Compactness: 10
- **Ventaja**: Reduce complejidad preservando bordes
- **Uso**: Preprocesamiento para otros algoritmos
- **Métricas**:
  - Número de superpixels generados
  - Nivel de compactness
  - Tiempo de ejecución

---

### 🎬 **Procesamiento de Videos**

#### 1. **Detección de Movimiento Espacial** 🏃
```python
process_video_motion(video_path, progress_callback)
```
- **Algoritmo**: Background Subtraction (MOG2)
- **Función**: Detecta objetos en movimiento en el dominio espacial
- **Proceso**:
  1. Modelado del fondo estático
  2. Sustracción de fondo
  3. Operaciones morfológicas (limpieza de ruido)
  4. Umbralización binaria
- **Visualización**: Áreas con movimiento en **rojo**
- **Parámetros**:
  - History: 500 frames
  - Variance threshold: 16
  - Sin detección de sombras
- **Limitación**: Procesa los primeros 50 frames
- **Uso**: Vigilancia, tracking de objetos
- **Salida**: Video procesado en formato AVI

#### 2. **Análisis de Movimiento en Frecuencia** 📊
```python
process_video_frequency(video_path, progress_callback)
```
- **Algoritmo**: FFT (Fast Fourier Transform) temporal
- **Función**: Analiza patrones de movimiento en el dominio de frecuencia
- **Proceso**:
  1. Conversión a escala de grises
  2. FFT en la dimensión temporal
  3. Cálculo de magnitud del espectro
  4. Detección de frecuencias de movimiento
  5. Normalización y umbralización
- **Visualización**: Zonas de movimiento frecuencial en **verde**
- **Ventaja**: Detecta movimientos periódicos y patrones
- **Limitación**: Procesa los primeros 50 frames
- **Uso**: Análisis de vibraciones, detección de patrones cíclicos
- **Salida**: Video procesado en formato AVI

---

### 📊 **Funciones Auxiliares**

#### Comparación de Técnicas
```python
compare_all_techniques(image)
```
- **Función**: Ejecuta todas las técnicas y compara rendimiento
- **Métricas comparadas**:
  - Tiempo de ejecución
  - Clasificación de eficiencia (Rápida/Media/Lenta)
- **Salida**: DataFrame de Pandas con resultados
- **Uso**: Selección de algoritmo óptimo para casos específicos

---

## 📊 Estructura del Proyecto

```
proyecto/
│
├── app.py                  # Aplicación principal Streamlit
├── requirements.txt        # Dependencias del proyecto
├── README.md              # Este archivo
│
├── venv/                  # Entorno virtual (no incluir en git)
│
└── temp/                  # Videos procesados temporales (generados automáticamente)
```

---

## 🎨 Interfaz de Usuario

### Panel de Control
- **Tab Imagen**: Carga y procesa imágenes estáticas
- **Tab Video**: Carga y analiza videos

### Visualización
- **Área principal**: Muestra imagen/video procesado
- **Métricas en tiempo real**: Tarjetas con estadísticas
- **Comparación**: Tabla comparativa de todas las técnicas

### Controles
- **8 Botones de técnicas**: Aplica algoritmos individuales
- **Restaurar**: Vuelve a la imagen original
- **Comparar**: Ejecuta análisis completo
- **Limpiar Video**: Elimina video procesado
- **Descargar**: Exporta video procesado

---

## 💡 Casos de Uso

### Imágenes
1. **Análisis de documentos**: Umbralización adaptativa
2. **Reconocimiento de objetos**: Detección de puntos (ORB)
3. **Análisis arquitectónico**: Detección de líneas
4. **Segmentación médica**: Watershed por regiones
5. **Preprocesamiento**: Superpixels SLIC

### Videos
1. **Vigilancia**: Detección de movimiento espacial
2. **Análisis deportivo**: Tracking de movimientos
3. **Control de calidad**: Detección de anomalías
4. **Análisis de vibraciones**: Dominio de frecuencia

---

## ⚙️ Configuración Avanzada

### Modificar Parámetros

Para ajustar los parámetros de los algoritmos, edita las funciones en `app.py`:

```python
# Ejemplo: Cambiar número de puntos ORB
orb = cv2.ORB_create(
    nfeatures=2000,  # Cambiar de 1000 a 2000
    scaleFactor=1.2,
    nlevels=8
)
```

### Cambiar Puerto de Streamlit

```bash
streamlit run app.py --server.port 8080
```

---

## 🐛 Solución de Problemas

### Error: "No module named 'cv2'"
```bash
pip install opencv-python
```

### Error: "DLL load failed" (Windows)
```bash
pip uninstall opencv-python
pip install opencv-python-headless
```

### Video no se procesa
- Verifica que el formato sea MP4, AVI o MOV
- Asegúrate de que el video no esté corrupto
- Los videos muy largos se procesan solo los primeros 50 frames

---

## 📈 Rendimiento

| Técnica | Velocidad | Uso de Memoria | Mejor Para |
|---------|-----------|----------------|------------|
| Puntos (ORB) | ⚡ Rápida | Bajo | Matching de features |
| Líneas (Hough) | 🔄 Media | Medio | Estructuras geométricas |
| Bordes (Canny) | ⚡ Rápida | Bajo | Contornos |
| Combinada | ⏱️ Lenta | Alto | Análisis completo |
| OTSU | ⚡ Rápida | Bajo | Binarización global |
| Adaptativa | 🔄 Media | Medio | Iluminación variable |
| Regiones (Watershed) | ⏱️ Lenta | Alto | Segmentación precisa |
| Superpixels (SLIC) | 🔄 Media | Medio | Preprocesamiento |

---

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Para cambios importantes:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

---

## 📝 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo LICENSE para más detalles.

---

## 👥 Autores

- **Tu Nombre** - [Tu GitHub](https://github.com/tu-usuario)

---

## 🙏 Agradecimientos

- OpenCV Community
- scikit-image Team
- Streamlit Developers
- Computer Vision Research Community

---

## 📚 Referencias

- [OpenCV Documentation](https://docs.opencv.org/)
- [scikit-image Documentation](https://scikit-image.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Canny Edge Detection Paper](https://ieeexplore.ieee.org/document/4767851)
- [SLIC Superpixels Paper](https://infoscience.epfl.ch/record/177415)

---

## 🔮 Roadmap Futuro

- [ ] Más algoritmos de segmentación (Mask R-CNN, U-Net)
- [ ] Soporte para procesamiento en GPU
- [ ] API REST para integración
- [ ] Modo batch para múltiples archivos
- [ ] Exportación en más formatos
- [ ] Análisis de video completo (sin límite de frames)
- [ ] Integración con modelos de deep learning

---

**⭐ Si te gustó este proyecto, dale una estrella en GitHub!**
