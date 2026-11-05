"""
Motor de Inferencia YOLOv11n (`best.pt`) y Módulo de Alertas en Tiempo Real (Requisitos 1 y 2).
Carga el modelo entrenado `best.pt` (10 clases: Fresh/Rotten de Carrot, Cucumber, Pepper, Potato, Tomato),
ejecuta la detección sobre la imagen previamente anonimizada en origen, dibuja las cajas delimitadoras
con código de color (Verde = Fresco, Rojo = Podrido/Alerta), y dispara la notificación automática (RF-02).
"""

import os
import base64
from typing import Dict, Any, List, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = os.environ.get(
    "FRESHGUARD_MODEL_PATH",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "best.pt"),
)

# Mapa oficial de las 10 clases entrenadas en best.pt
CLASS_NAMES = {
    0: "Fresh_Carrot",
    1: "Fresh_Cucumber",
    2: "Fresh_Pepper",
    3: "Fresh_Potato",
    4: "Fresh_Tomato",
    5: "Rotten_Carrot",
    6: "Rotten_Cucumber",
    7: "Rotten_Pepper",
    8: "Rotten_Potato",
    9: "Rotten_Tomato",
}

SPANISH_LABELS = {
    "Fresh_Carrot": "Zanahoria Fresca",
    "Fresh_Cucumber": "Pepino Fresco",
    "Fresh_Pepper": "Pimiento Fresco",
    "Fresh_Potato": "Papa Fresca",
    "Fresh_Tomato": "Tomate Fresco",
    "Rotten_Carrot": "Zanahoria Podrida",
    "Rotten_Cucumber": "Pepino Podrido",
    "Rotten_Pepper": "Pimiento Podrido",
    "Rotten_Potato": "Papa Podrida",
    "Rotten_Tomato": "Tomate Podrido",
}


class ProduceDetector:
    """Singleton wrapper alrededor de YOLO('best.pt')."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = YOLO(MODEL_PATH)
        return cls._instance

    def predict_image(
        self,
        image_bgr: np.ndarray,
        conf_threshold: float = 0.25,
        aisle_zone: str = "Anaquel Perecederos A1",
    ) -> Tuple[List[Dict[str, Any]], str, bool, str]:
        """
        Ejecuta inferencia real con `best.pt` sobre `image_bgr`.
        Retorna:
          - detections: lista de objetos detectados con clase, estado ('Fresh'/'Rotten'), confianza y bbox
          - annotated_b64: imagen anotada en formato Base64 (JPEG)
          - alert_triggered: True si al menos 1 producto está 'Rotten'
          - alert_message: Mensaje de notificación enviado al encargado de piso
        """
        results = self.model.predict(source=image_bgr, conf=conf_threshold, verbose=False)
        res = results[0]

        annotated = image_bgr.copy()
        detections: List[Dict[str, Any]] = []
        rotten_items: List[str] = []

        if res.boxes is not None and len(res.boxes) > 0:
            boxes_xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            cls_ids = res.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cid in zip(boxes_xyxy, confs, cls_ids):
                cname = self.model.names.get(int(cid), CLASS_NAMES.get(int(cid), f"Class_{cid}"))
                parts = cname.split("_", 1)
                condition_state = parts[0] if len(parts) > 1 else ("Rotten" if "Rotten" in cname else "Fresh")
                produce_type = parts[1] if len(parts) > 1 else cname

                x1, y1, x2, y2 = [int(round(v)) for v in box]
                is_rotten = condition_state.lower() == "rotten"

                color = (40, 40, 230) if is_rotten else (50, 200, 60)
                label_es = SPANISH_LABELS.get(cname, cname)
                caption = f"{label_es} ({conf * 100:.1f}%)"

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
                cv2.rectangle(annotated, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), color, -1)
                cv2.putText(
                    annotated,
                    caption,
                    (x1 + 4, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                if is_rotten:
                    rotten_items.append(f"{label_es} ({conf * 100:.0f}%)")

                detections.append(
                    {
                        "class_id": int(cid),
                        "class_name": cname,
                        "label_es": label_es,
                        "produce_type": produce_type,
                        "condition_state": "Rotten" if is_rotten else "Fresh",
                        "confidence": float(conf),
                        "bbox": [x1, y1, x2, y2],
                        "aisle_zone": aisle_zone,
                    }
                )

        alert_triggered = len(rotten_items) > 0
        if alert_triggered:
            alert_message = (
                f"[ALERTA RETIRO INMEDIATO - {aisle_zone}] Se detectaron {len(rotten_items)} "
                f"producto(s) en deterioro: {', '.join(rotten_items)}. Notificación enviada a terminal de piso."
            )
            # Banner superior de alerta en la imagen
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 32), (25, 25, 210), -1)
            cv2.putText(
                annotated,
                f"ALERTA DE MERMA ACTIVA: {len(rotten_items)} PRODUCTO(S) DETERIORADO(S)",
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            alert_message = f"Sin alertas en {aisle_zone}. Productos inspeccionados en estado óptimo."

        _, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        annotated_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

        return detections, annotated_b64, alert_triggered, alert_message
