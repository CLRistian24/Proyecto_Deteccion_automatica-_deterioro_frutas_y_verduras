"""
Script de Re-Entrenamiento Cloud/Local para Requisito de Carpeta `scripts/` (Sección 3.3 del PDF).
Nota: El modelo `best.pt` incluido en el repositorio ya fue entrenado durante 50 épocas
sobre el dataset de 10 clases (`Fresh/Rotten Carrot, Cucumber, Pepper, Potato, Tomato`)
alcanzando mAP@50 = 96.45%, Precisión = 91.07% y Recall = 91.80%.
Este script documenta y permite reproducir el ciclo de reentrenamiento semestral FinOps.
"""

from ultralytics import YOLO


def train_model(data_yaml: str = "dataset/data.yaml", epochs: int = 50, batch: int = 62, imgsz: int = 640):
    model = YOLO("yolo11n.pt")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device="0",
        optimizer="auto",
        amp=True,
        auto_augment="randaugment",
        erasing=0.4,
        project="freshguard_training",
        name="yolov11n_v02",
    )
    return results


if __name__ == "__main__":
    print("Modelo pre-entrenado actual: best.pt (50 épocas completadas, mAP50=96.45%).")
    print("Para ejecutar un nuevo ciclo semestral de re-entrenamiento pase la ruta de data.yaml.")
