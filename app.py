from flask import Flask, render_template, request, send_file, redirect, url_for, session
import os
import cv2
from werkzeug.utils import secure_filename
from scanner import detectar_bordes
from PIL import Image
import time
from uuid import uuid4

# === Rutas absolutas basadas en la ubicación del archivo ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'output')

# === Inicialización de Flask ===
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'clave_secreta_segura'  # Necesaria para usar session

# === Crear carpetas si no existen ===
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === Página principal: limpia archivos previos y muestra el index ===
@app.route('/')
def index():
    for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
        for f in os.listdir(folder):
            os.remove(os.path.join(folder, f))
    session.clear()
    return render_template('index.html')

# === Procesar imágenes subidas, generar PDF, y redirigir al resultado ===
@app.route('/procesar', methods=['POST'])
def procesar():
    imagenes = request.files.getlist('imagenes')
    procesadas = []
    tamanos = []

    if not imagenes:
        return "No se recibieron imágenes", 400

    # Procesar cada imagen
    for img in imagenes:
        filename = secure_filename(img.filename)
        ruta = os.path.join(UPLOAD_FOLDER, filename)
        img.save(ruta)

        salida = detectar_bordes(ruta)
        nombre_salida = os.path.join(OUTPUT_FOLDER, filename)
        cv2.imwrite(nombre_salida, salida)
        procesadas.append(nombre_salida)
        tamanos.append(salida.shape[:2])  # (alto, ancho)

    # Calcular tamaño mediano (basado en área más cercana a la mediana)
    areas = [w*h for (w, h) in tamanos]
    idx_mediana = sorted(range(len(areas)), key=lambda i: areas[i])[len(areas) // 2]
    target_size = tamanos[idx_mediana][::-1]  # PIL usa (ancho, alto)

    imagenes_rgb = []
    for p in procesadas:
        img = Image.open(p).convert("RGB")
        img.thumbnail(target_size, Image.Resampling.LANCZOS)

        # Centrar la imagen en un lienzo del tamaño target_size
        fondo = Image.new("RGB", target_size, (255, 255, 255))
        offset_x = (target_size[0] - img.width) // 2
        offset_y = (target_size[1] - img.height) // 2
        fondo.paste(img, (offset_x, offset_y))
        imagenes_rgb.append(fondo)

    # Generar PDF con nombre único
    timestamp = int(time.time())
    pdf_name = f"documento_{timestamp}_{uuid4().hex[:6]}.pdf"
    pdf_path = os.path.join(OUTPUT_FOLDER, pdf_name)

    imagenes_rgb[0].save(pdf_path, save_all=True, append_images=imagenes_rgb[1:])

    # Guardar nombre del PDF en la sesión
    session['pdf_name'] = pdf_name
    print(f"[INFO] PDF generado en: {os.path.abspath(pdf_path)}")
    return redirect(url_for('resultado'))

# === Mostrar el visor del PDF generado ===
@app.route('/resultado')
def resultado():
    pdf_name = session.get('pdf_name')
    if not pdf_name:
        return redirect(url_for('index'))
    return render_template('resultado.html', timestamp=int(time.time()))

# === Servir el PDF embebido en el navegador ===
@app.route('/ver_pdf')
def ver_pdf():
    pdf_name = session.get('pdf_name')
    if not pdf_name:
        return "PDF no disponible en la sesión.", 404

    pdf_path = os.path.join(OUTPUT_FOLDER, pdf_name)
    if not os.path.isfile(pdf_path):
        print(f"[ERROR] Archivo no encontrado: {pdf_path}")
        return "Archivo PDF no encontrado en disco.", 404

    print(f"[INFO] Mostrando PDF: {pdf_path}")
    return send_file(pdf_path, mimetype='application/pdf')

# === Permitir descarga directa del PDF ===
@app.route('/descargar_pdf')
def descargar_pdf():
    pdf_name = session.get('pdf_name')
    if not pdf_name:
        return redirect(url_for('index'))

    pdf_path = os.path.join(OUTPUT_FOLDER, pdf_name)
    if not os.path.isfile(pdf_path):
        print(f"[ERROR] No se puede descargar. Archivo no existe: {pdf_path}")
        return "Archivo PDF no disponible para descargar.", 404

    print(f"[INFO] Descargando PDF: {pdf_path}")
    return send_file(pdf_path, as_attachment=True)

# === Ruta para reiniciar: limpia todo y vuelve al inicio ===
@app.route('/reiniciar')
def reiniciar():
    for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
        for f in os.listdir(folder):
            os.remove(os.path.join(folder, f))
    session.clear()
    return redirect(url_for('index'))

# === Ejecutar servidor ===
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
