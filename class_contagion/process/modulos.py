import numpy as np

from datetime import datetime
import time
import os

import cv2 as cv

from configuration.config import OUTPUT_DIR, MODEL_DIR


out_archivo = None
def aux_saving(grabando,tiempo_archivo,tiempo_reinicio_video,nombre_archivo,\
        fourcc,milisegundos,frame_width,frame_height,\
        id_sospechoso,indi,t_inicio):
    
    '''This function saves the video a few seconds before that the event occurs.
    Also resets the time before the event starts and deletes the video if it is not necessary.'''

    global out_archivo
    if (not grabando)and (tiempo_archivo>tiempo_reinicio_video):
        if f'{nombre_archivo}.mp4' in os.listdir(f'{MODEL_DIR}/'):
#            if ('out_archivo' in locals() or 'out_archivo' in globals()):
            if out_archivo is not None:
                out_archivo.release()
#                del(out_archivo)
                out_archivo = None
            os.remove(f'{MODEL_DIR}/{nombre_archivo}.mp4')
        nombre_archivo = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        out_archivo = cv.VideoWriter(f'{OUTPUT_DIR}/{nombre_archivo}.mp4',fourcc,(1000/milisegundos),(frame_width,frame_height))
        
#        print(f"DENTRO:  {'out_archivo' in locals() or 'out_archivo' in globals()}")

        return grabando,0,nombre_archivo,\
                1, time.time()
    elif grabando and (id_sospechoso is None) and indi==1:
        out_archivo.release()
        #del(out_archivo)
        out_archivo = None


#        print(f"DENTRO:  {'out_archivo' in locals() or 'out_archivo' in globals()}")

        return False,tiempo_reinicio_video+0.1,None,\
                indi,t_inicio
       
    elif tiempo_archivo<=tiempo_reinicio_video:
        
#        print(f"DENTRO:  {'out_archivo' in locals() or 'out_archivo' in globals()}")

        return grabando,time.time()-t_inicio,nombre_archivo,\
                indi,t_inicio
    

def event_association(cls_tg,r,
                      ntarget,nnotarget,
                      id_sospechoso,lista_humanos, marca, b, annotator, milisegundos):
    
    '''This function associate the closest person to a given object given by the user.'''

    global centroide_tg

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
    
    return ntarget,nnotarget,id_sospechoso,lista_humanos, marca 


# def frameando():

# if out_archivo is not None:
#     out_archivo.write(frame)


# _, frame = video.read()
# results = model.track(frame, conf=0.6, save=False, show=False)


# for r in results:

#     annotator = Annotator(frame)

#     lista_humanos = ((r.boxes.cls == 0.).nonzero()).flatten()

#     b = r.boxes.xyxy[lista_humanos]
#     #c = r.boxes.cls[lista_humanos]

#     for h in b:
        
#         annotator.box_label(h, f'Android -- fr {nframe}', color=(255, 0, 0))
    
#     ntarget,nnotarget,id_sospechoso,lista_humanos, marca =\
#     event_association(cls_tg,r,
#                 ntarget,nnotarget,
#                 id_sospechoso,lista_humanos, marca, b, annotator, milisegundos)
