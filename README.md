# YOLOv11: Detección del estado de la fruta y verdura en tiempo real

Este repositorio contiene la implementación de inferencia local de un modelo YOLOv11 entrenado para detectar el estado de la fruta y verdura (descomposición y normal) mediante una cámara web en tiempo real.

## Requisitos del Sistema

*   **Sistema Operativo:** Fedora 44
*   **Python:** 3.11+
*   **Hardware:** Webcam estándar (`/dev/video0`).
*   **No se requiere una GPU dedicada para la inferencia en tiempo real, es suficiente con gráficos integrados**

## Instalación

1. Clonar el repositorio:
```bash
git clone [https://github.com/](https://github.com/)[Tu-Usuario]/[Tu-Repo].git
cd [Tu-Repo]
```

2. Crear y activar el entorno virtual:
```bash
python -m venv venv

# En Windows:
venv\Scripts\activate
# En Linux/macOS:
source venv/bin/activate
```

3. Instalar las dependencias:
```bash
pip install -r requirements.txt
```
*Nota: Si se utiliza GPU, instalar los binarios de PyTorch correspondientes a la versión de CUDA instalada localmente antes de ejecutar las dependencias del archivo `requirements.txt`.*

## Configuración de Pesos (Weights)

Descargar el archivo del modelo entrenado (`best.pt`) desde [Insertar enlace de descarga, ej. Google Drive / GitHub Releases] y colocarlo en el directorio raíz. La estructura del proyecto debe ser estrictamente la siguiente:

```text
/tu-repo
  ├── best.pt
  ├── [nombre_de_tu_script].py
  ├── requirements.txt
  └── README.md
```

## Ejecución

Para iniciar el flujo de video y la ejecución del modelo de inferencia, ejecutar en la terminal:

```bash
python [nombre_de_tu_script].py
```

Presionar la tecla `q` en la ventana de visualización para liberar el búfer de la cámara y terminar el proceso de forma segura.
