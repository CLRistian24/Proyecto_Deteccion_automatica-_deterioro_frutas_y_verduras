from ultralytics import YOLO
#Cargar el modelo entrenado
model = YOLO("best.pt")
results = model.predict(source="0", show=True)
