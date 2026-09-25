# YOLOv11: Sistema de Detección del Estado de Frutas y Verduras en Tiempo Real

Este repositorio contiene la implementación de inferencia local de un modelo YOLOv11 entrenado para clasificar el estado de frutas y verduras (normal o en descomposición) mediante captura de video en tiempo real.

## Requisitos del Sistema

*   **Sistema Operativo:** Probado y validado en Fedora 44.
*   **Python:** 3.11 o superior.
*   **Hardware:** Webcam estándar (dispositivo `/dev/video0`).
*   **Aceleración de Hardware:** No requiere GPU dedicada. La inferencia es ejecutable en CPU o gráficos integrados manteniendo tasas de procesamiento funcionales.

## Instalación

Es obligatorio utilizar un entorno virtual para evitar conflictos de dependencias con los paquetes del sistema operativo.

1. Clonar el repositorio:
```bash
git clone [https://github.com/CLRistian24/Proyecto_Deteccion_automatica-_deterioro_frutas_y_verduras.git](https://github.com/CLRistian24/Proyecto_Deteccion_automatica-_deterioro_frutas_y_verduras.git)
cd Proyecto_Deteccion_automatica-_deterioro_frutas_y_verduras
```

2. Crear y activar un entorno virtual:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Instalar la dependencia principal (`ultralytics` instalará automáticamente dependencias subyacentes requeridas como `torch` y `opencv-python`):
```bash
pip install ultralytics
```

## Configuración de Pesos (Weights)

Ubica el archivo del modelo entrenado (`best.pt`) y  verifica que esté en el directorio raíz. 

La estructura del directorio debe ser exactamente la siguiente:

```text
/Proyecto_Deteccion_automatica-_deterioro_frutas_y_verduras
  ├── best.pt
  ├── inferencia.py
  └── README.md
```

## Ejecución

Para iniciar el flujo de video y la ejecución del modelo de inferencia, ejecutar en la terminal:

```bash
python inferencia.py
```

Presionar la tecla `q` en la ventana de visualización para liberar el búfer de la cámara y terminar el proceso del script de forma segura.

##Video
A continuación te compartimos un link con un video que presenta el modelo en acción: https://youtu.be/u9eHYYvYQwc
