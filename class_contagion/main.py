from pathlib import Path
import sys
#sys.path.append(Path(__file__).resolve().parent.parent)

#print(sys.path)

import cv2 as cv
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import sys
import numpy as np
import os
from datetime import datetime
import time

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse



from process.modelo import modelo

##########################
# PARAMETROS

# Permitamos a opencv a usar la GPU
os.environ['OPENCV_DNN_OPENCL_ALLOW_ALL_DEVICES'] = '1'

milisegundos = int(sys.argv[1])
cls_tg = float(sys.argv[2])
prueba = sys.argv[3]
tiempo_reinicio_video = 3 # Segundos que tienen que transcurrir para que el video que captura el momento se reinicie si no ha capturado nada.

##########################
# APP
# Para levantrar la aplicación: uvicorn main:app --reload
app = FastAPI(title='Basic WebCam Test',
              description='Trying to deploy a webcam service')
html = """
<!DOCTYPE html>
<html>
    <head>
        <title>WebCam</title>
    </head>
    <body>
        <h1>Streaming WebCam</h1>
        <div>
        Para empezar la transmisión ir al endpoint: <strong><a href='/webcam'>Webcam</a></strong>
        </div>
        <div>
        Para detener la transmisión ir al endpoint: <strong><a href='/stop'>Stop</a></strong>
        </div>
 
    </body>
</html>
"""

@app.get("/")
async def index():
    return HTMLResponse(html)

@app.get("/webcam")
def webcam():
    # Devolver la transmisión de imágenes como un video.
    return StreamingResponse(modelo(milisegundos, cls_tg, prueba, tiempo_reinicio_video), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/stop")
def stop():
    # Detenemos la transmisión.
    global transmite
    transmite = False
    return {"mensaje": "Transmisión detenida"}

if __name__=='__main__':
    import uvicorn
    #uvicorn.run(app, host="localhost", port=8001)
    #uvicorn.run(app, host="0.0.0.0", port=8001)