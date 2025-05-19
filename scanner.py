import cv2
import numpy as np
remove = None

from PIL import Image
import io
import os
import urllib.request

# Descargar modelo si no existe
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "u2netp.onnx")

if not os.path.exists(MODEL_PATH):
    print("[INFO] Descargando modelo u2net.onnx...")
    url = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("[INFO] Modelo descargado en:", MODEL_PATH)

def aplicar_laplaciano(imagen):
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])
    return cv2.filter2D(imagen, -1, kernel)

def eliminar_fondo_con_ia(imagen_bgr):
    global remove
    if remove is None:
        from rembg import new_session
        session = new_session(MODEL_PATH)
        from rembg import remove as base_remove
        remove = lambda input_image: base_remove(input_image, session=session)

    imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(imagen_rgb)
    pil_output = remove(pil_image)
    resultado_np = np.array(pil_output)
    if resultado_np.shape[2] == 4:
        resultado_np = cv2.cvtColor(resultado_np, cv2.COLOR_RGBA2BGR)
    else:
        resultado_np = cv2.cvtColor(resultado_np, cv2.COLOR_RGB2BGR)
    return resultado_np

def detectar_bordes(ruta):
    img = cv2.imread(ruta)
    img = eliminar_fondo_con_ia(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edged = cv2.Canny(blurred, 85, 200)
    edged = cv2.dilate(edged, None, iterations=2)
    edged = cv2.erode(edged, None, iterations=1)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    document_contour = None
    max_area = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area > max_area:
            max_area = area
            document_contour = c

    if document_contour is not None:
        peri = cv2.arcLength(document_contour, True)
        approx = cv2.approxPolyDP(document_contour, 0.02 * peri, True)

        if len(approx) == 4:
            pts = np.array([p[0] for p in approx], dtype="float32")
        else:
            rect = cv2.minAreaRect(document_contour)
            box = cv2.boxPoints(rect)
            pts = np.array(box, dtype="float32")

        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

        h_ratio = warped.shape[0] / img.shape[0]
        w_ratio = warped.shape[1] / img.shape[1]
        if h_ratio > 1.2 or w_ratio > 1.2:
            x, y, w, h = cv2.boundingRect(document_contour)
            warped = img[y:y+h, x:x+w]

        gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        if warped.shape[0] > 20 and warped.shape[1] > 20:
            if np.mean(gray_warped[:10, :]) > 200 and np.mean(gray_warped[-10:, :]) > 200:
                warped = warped[10:-10, :]
            if np.mean(gray_warped[:, :10]) > 200 and np.mean(gray_warped[:, -10:]) > 200:
                warped = warped[:, 10:-10]

        lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        warped_equalizado = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

        laplacian = aplicar_laplaciano(warped_equalizado)
        laplacian_abs = cv2.convertScaleAbs(laplacian)
        warped_f = warped_equalizado.astype(np.float32)
        laplacian_f = laplacian_abs.astype(np.float32)
        sharpened = np.clip(warped_f - laplacian_f, 0, 255).astype(np.uint8)
        ajustada = cv2.convertScaleAbs(sharpened, alpha=1.5, beta=0)
        return ajustada

    return img

# Requisitos:
# pip install opencv-python numpy pillow rembg onnxruntime
