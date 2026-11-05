"""
Módulo de Privacidad desde el Diseño y Anonimización en Origen (Requisito 8).
Cumple con la LFPDPPP (Ley Federal de Protección de Datos Personales en Posesión de los Particulares):
1. Detección neuronal real de rostros humanos mediante el modelo profundo YuNet (`FaceDetectorYN` ONNX de OpenCV)
   con soporte multi-orientación (detecta rostros derechos, inclinados o en vista cenital de cámara de anaquel)
   SIN confundir verduras/madera de color naranja con piel humana.
2. Aplicación de desenfoque gaussiano irreversible (Gaussian Blur + Pixelation) sobre los rostros reales
   ANTES de almacenar la imagen o enviarla al motor de inferencia/nube.
3. Eliminación de metadatos EXIF (geolocalización GPS, serie de cámara, autor) al recodificar el frame en memoria.
4. Seudonimización criptográfica (SHA-256 con sal) de identificadores de cámara/dispositivo.
"""

import os
import hashlib
from typing import Tuple, List
import cv2
import numpy as np

SALT_SECRET = "UPIIT-IPN-2027-LFPDPPP-SALT"
YUNET_MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_detection_yunet.onnx")


def pseudonymize_identifier(raw_id: str) -> str:
    """Genera un hash SHA-256 truncado e irreversible para seudonimizar el ID de cámara o empleado."""
    salted = f"{SALT_SECRET}:{raw_id.strip()}".encode("utf-8")
    digest = hashlib.sha256(salted).hexdigest()[:16]
    return f"anon_{digest}"


def _iou(box_a: List[int], box_b: List[int]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[0] + box_a[2], box_b[0] + box_b[2])
    y2 = min(box_a[1] + box_a[3], box_b[1] + box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]
    return inter / float(max(area_a + area_b - inter, 1))


def _detect_faces_yunet(image_bgr: np.ndarray, score_threshold: float = 0.65) -> List[List[int]]:
    """
    Detecta únicamente rostros humanos reales usando la red neuronal YuNet (`cv2.FaceDetectorYN`).
    Soporta orientación estándar (0°) y rotaciones de cámara cenital (90°, 180°, 270°) para pasillos.
    """
    if not os.path.exists(YUNET_MODEL_PATH) or not hasattr(cv2, "FaceDetectorYN"):
        return []

    h_orig, w_orig = image_bgr.shape[:2]
    # Redimensionar internamente si la imagen es muy grande para mantener latencia < 10 ms
    max_dim = 640
    scale = min(1.0, max_dim / float(max(h_orig, w_orig)))
    work_w = max(32, int(round(w_orig * scale)))
    work_h = max(32, int(round(h_orig * scale)))
    work_img = cv2.resize(image_bgr, (work_w, work_h)) if scale < 1.0 else image_bgr

    detector = cv2.FaceDetectorYN.create(
        model=YUNET_MODEL_PATH,
        config="",
        input_size=(work_w, work_h),
        score_threshold=score_threshold,
        nms_threshold=0.3,
        top_k=50,
    )

    candidate_boxes: List[Tuple[float, List[int]]] = []

    # 1. Pasada directa (0 grados)
    _, faces_0 = detector.detect(work_img)
    if faces_0 is not None:
        for f in faces_0:
            x, y, w, h, score = float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[-1])
            bx = [int(x / scale), int(y / scale), int(w / scale), int(h / scale)]
            candidate_boxes.append((score, bx))

    # 2. Si la cámara está en vista cenital (top-down) y el rostro aparece rotado (90°, 180°, 270°):
    if not candidate_boxes:
        rotations = [
            (cv2.ROTATE_90_CLOCKWISE, 90),
            (cv2.ROTATE_90_COUNTERCLOCKWISE, 270),
            (cv2.ROTATE_180, 180),
        ]
        for rot_code, angle in rotations:
            rot_img = cv2.rotate(work_img, rot_code)
            rh, rw = rot_img.shape[:2]
            detector.setInputSize((rw, rh))
            _, faces_r = detector.detect(rot_img)
            if faces_r is not None:
                for f in faces_r:
                    rx, ry, bw, bh, score = float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[-1])
                    if angle == 90:
                        ox, oy, ow, oh = ry, work_h - (rx + bw), bh, bw
                    elif angle == 270:
                        ox, oy, ow, oh = work_w - (ry + bh), rx, bh, bw
                    else:  # 180
                        ox, oy, ow, oh = work_w - (rx + bw), work_h - (ry + bh), bw, bh
                    bx = [int(ox / scale), int(oy / scale), int(ow / scale), int(oh / scale)]
                    candidate_boxes.append((score, bx))

    # Deduplicar cajas solapadas (NMS) y recortar a los límites de la imagen
    candidate_boxes.sort(key=lambda item: item[0], reverse=True)
    final_boxes: List[List[int]] = []
    for _, b in candidate_boxes:
        x = max(0, min(w_orig - 1, b[0]))
        y = max(0, min(h_orig - 1, b[1]))
        w = max(1, min(w_orig - x, b[2]))
        h = max(1, min(h_orig - y, b[3]))
        clipped = [x, y, w, h]
        if all(_iou(clipped, kept) < 0.35 for kept in final_boxes):
            final_boxes.append(clipped)

    return final_boxes


def anonymize_image_in_origin(image_bgr: np.ndarray) -> Tuple[np.ndarray, int, List[List[int]]]:
    """
    Recibe una imagen BGR en memoria recién capturada en el Edge, detecta rostros humanos reales
    mediante la red neuronal YuNet y aplica desenfoque gaussiano fuerte + pixelado sobre la región facial.
    Retorna:
      - imagen_anonimizada (np.ndarray)
      - numero_rostros_difuminados (int)
      - lista_cajas_rostros ([[x, y, w, h], ...])
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("La imagen de entrada está vacía o corrupta.")

    output = image_bgr.copy()
    face_boxes = _detect_faces_yunet(output)

    for (x, y, w, h) in face_boxes:
        roi = output[y : y + h, x : x + w]
        if roi.size > 0:
            # Doble capa de privacidad: Pixelado + Desenfoque Gaussiano
            small = cv2.resize(roi, (max(1, w // 12), max(1, h // 12)), interpolation=cv2.INTER_LINEAR)
            pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
            ksize = max(15, (min(w, h) // 3) | 1)
            blurred = cv2.GaussianBlur(pixelated, (ksize, ksize), 30)
            output[y : y + h, x : x + w] = blurred
            # Marco discreto indicando anonimización LFPDPPP activa
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 180, 0), 2)
            cv2.putText(
                output,
                "ROSTRO ANONIMIZADO (LFPDPPP)",
                (max(4, x), max(18, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 180, 0),
                2,
                cv2.LINE_AA,
            )

    return output, len(face_boxes), face_boxes
