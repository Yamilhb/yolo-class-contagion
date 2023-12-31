
# Librerías de sistema.
# from pathlib import Path
# import sys
# import os

# Librerías de imágenes.
import cv2 as cv
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

# Librerías misceláneas.
import time

# Librerías de la app.
from configuration.config import  OUTPUT_DIR, MODEL_DIR
from process.modulos import aux_saving, event_association, out_archivo

def modelo(milisegundos, cls_tg, prueba, tiempo_reinicio_video):

    print('ENTRE A LA FUNCIÓN')

    # Preparación del modelo.
    video = cv.VideoCapture(0)
    frame_width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))

    model = YOLO(f'{MODEL_DIR}/yolov8n.pt')  # load an official model
    model.to('cuda')

    fourcc  = cv.VideoWriter_fourcc(*'MP4V') 
    out = cv.VideoWriter(f'{OUTPUT_DIR}/prueba{prueba}.mp4',fourcc,(1000/milisegundos),(frame_width,frame_height), True)

    # Inicialización de parámetros.    
    nframe = 0
    ntarget = 0
    nnotarget = 0
    id_sospechoso = None
    grabando = False
    tiempo_archivo = 0
    t_inicio = time.time()
    nombre_archivo = None
    indi = None
    marca = None
    transmite = True
#    del(out_archivo)

###########
    while transmite:
#        print('\n\n'+'-+'*15,f'FRAME: {nframe}',f"   HORA: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        grabando,tiempo_archivo,nombre_archivo,indi, t_inicio = aux_saving(grabando,tiempo_archivo,tiempo_reinicio_video,nombre_archivo,\
                                                                                    fourcc,milisegundos,frame_width,frame_height,\
                                                                                id_sospechoso,indi,t_inicio)
        
        if out_archivo is not None:
            out_archivo.write(frame)


        ret, frame = video.read()
        results = model.track(frame, conf=0.6, save=False, show=False)


        for r in results:

            annotator = Annotator(frame)

            lista_humanos = ((r.boxes.cls == 0.).nonzero()).flatten()

            b = r.boxes.xyxy[lista_humanos]
            #c = r.boxes.cls[lista_humanos]

            for h in b:
                
               annotator.box_label(h, f'Android -- fr {nframe}', color=(255, 0, 0))
            
            ntarget,nnotarget,id_sospechoso,lista_humanos, marca =\
            event_association(cls_tg,r,
                      ntarget,nnotarget,
                      id_sospechoso,lista_humanos, marca, b, annotator, milisegundos)
 
        nframe += 1
        frame = annotator.result()


        out.write(frame)
        cv.imshow('YoL0v8',frame)
#        if cv.waitKey(milisegundos)& 0xFF==ord('q'): 
#            break

        if ret:
            print(f"ESTO: {len(cv.imencode('.jpg', frame))} :ES")
            _, buffer = cv.imencode('.jpg', frame)
            frame_bytes = (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            yield frame_bytes

    # RELEASE MEMORY
    ################
    out.release()
    video.release()
    cv.destroyAllWindows()

#modelo(milisegundos, cls_tg, prueba, tiempo_reinicio_video)