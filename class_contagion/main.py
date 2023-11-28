import cv2 as cv
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import sys
import numpy as np
import os
from datetime import datetime
import time
# Permitamos a opencv a usar la GPU
os.environ['OPENCV_DNN_OPENCL_ALLOW_ALL_DEVICES'] = '1'

milisegundos = int(sys.argv[1])
cls_tg = float(sys.argv[2])
prueba = sys.argv[3]
tiempo_reinicio_video = 3 # Segundos que tienen que transcurrir para que el video que captura el momento se reinicie si no ha capturado nada.


def main():
    video = cv.VideoCapture(0)
    frame_width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))

    model = YOLO('yolov8n.pt')  # load an official model
    model.to('cuda')

    fourcc  = cv.VideoWriter_fourcc(*'MP4V') 
    out = cv.VideoWriter(f'resultados/prueba{prueba}.mp4',fourcc,(1000/milisegundos),(frame_width,frame_height), True)
###########    
    nframe = 0
    ntarget = 0
    nnotarget = 0
    id_sospechoso = None
    grabando = False
    tiempo_archivo = 0
    t_inicio = time.time()
    nombre_archivo = None
    indi = None
###########
    while True:
        print('\n\n'+'-+'*15,f'FRAME: {nframe}',f"   HORA: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if (not grabando)and (tiempo_archivo>tiempo_reinicio_video):
            print('1------>',f'{nombre_archivo}.mp4','<------')
            print('111111:',os.listdir('resultados/'))
            if f'{nombre_archivo}.mp4' in os.listdir('resultados/'):
                if 'out_archivo' in locals() or 'out_archivo' in globals():
                    out_archivo.release()
                    del(out_archivo)
                os.remove(f'resultados/{nombre_archivo}.mp4')
            nombre_archivo = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print('2------>',nombre_archivo,'<------')
            print('222222:',os.listdir('resultados/'))
            out_archivo = cv.VideoWriter(f'resultados/{nombre_archivo}.mp4',fourcc,(1000/milisegundos),(frame_width,frame_height))
            tiempo_archivo = 0
            t_inicio = time.time()
            indi=1
        elif grabando and (id_sospechoso is None) and indi==1:
            out_archivo.release()
            del(out_archivo)
            nombre_archivo = None
            tiempo_archivo = tiempo_reinicio_video+0.1
            grabando = False
        elif tiempo_archivo<=tiempo_reinicio_video:
            tiempo_archivo = time.time()-t_inicio
        
        if 'out_archivo' in locals() or 'out_archivo' in globals():
            out_archivo.write(frame) 

        _, frame = video.read()
        results = model.track(frame, conf=0.6, save=False, show=False)

#        print('\n'+'-+'*15,f'FRAME: {nframe}')
                


        for r in results:

            annotator = Annotator(frame)

            lista_humanos = ((r.boxes.cls == 0.).nonzero()).flatten()

            b = r.boxes.xyxy[lista_humanos]
            #c = r.boxes.cls[lista_humanos]

            for h in b:
                
               annotator.box_label(h, f'Android -- fr {nframe}', color=(255, 0, 0))
            
            # Se detecta el evento?
            if cls_tg in r.boxes.cls:                
                # Buscamos evento.
                lista_tgs = ((r.boxes.cls == cls_tg).nonzero()).flatten()
                # Localizamos el centroide del evento target (Para después asociarlo al humano con centroide más cercano).
                centroide_tg = r.boxes.xywh[lista_tgs][0][:2]
                ntarget +=1
            
            # Condición auxiliar para no romper el flujo cuando el evento no se detecta en este frame pero sí en los anteriores (se está produciendo el evento).
            if ((cls_tg not in r.boxes.cls)and (ntarget>0)):
                ntarget +=1
                nnotarget +=1
            
            # Confirmamos que se produce el evento (se ha detectado en 3 frames cuasi-consecutivos): 
            #   - Se busca al causante del evento y se registra su id.
            if (ntarget==3) and (id_sospechoso is None):
                # Calculamos la distancia entre los humanos y el evento.
                distancias_h_tg = np.array([[x[0],
                                    np.linalg.norm(centroide_tg-x[1][:2])]
                                      for x in zip(r.boxes.id[lista_humanos],r.boxes.xywh[lista_humanos])])
                # Buscamos el ID del humano más cercano al evento.
                posicion_minima = np.argmin(distancias_h_tg[:,1])
                id_sospechoso = distancias_h_tg[posicion_minima][0]
            
            # Si el evento ya pasó, reseteo las variables que corresponde.
            if nnotarget>=3:
                ntarget = 0
                nnotarget = 0
                lista_tgs = None
                centroide_tg = None
                distancias_h_tg = None
                posicion_minima = None

            # Se busca al sospechoso y se graba la parte del video donde se captura el evento.
            if id_sospechoso is not None:
                if (r.boxes.id is not None)and (((r.boxes.id == id_sospechoso).sum())>0) and (0. in r.boxes.cls):
                    marca = 0
                    for h in b:
                        annotator.box_label(h, f'Contagion by a {r.names[int(cls_tg)]}',color = (0, 0, 255))
                    grabando = True
                else:
                    marca +=1
                    if marca >= (1000//milisegundos): # 'marca' nos indica los frame que lleva fuera de imagen el suspect
                        id_sospechoso = None # Si lleva más de 1 segundo, lo borramos.
 
        nframe += 1
        frame = annotator.result()

        
        out.write(frame)
        cv.imshow('YoL0v8',frame)
        if cv.waitKey(milisegundos)& 0xFF==ord('q'): 
            break

    # RELEASE MEMORY
    ################
    out.release()
    video.release()
    cv.destroyAllWindows()

if __name__=='__main__':
    main()