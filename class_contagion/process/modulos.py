
from datetime import datetime
import time
import os

import cv2 as cv

from config.config import OUTPUT_DIR, MODEL_DIR


def aux_saving(grabando,tiempo_archivo,tiempo_reinicio_video,nombre_archivo,\
        fourcc,milisegundos,frame_width,frame_height,\
        id_sospechoso,indi,t_inicio):
    '''This function saves the video a few seconds before that the event occurs.
    Also resets the time before the event starts and deletes the video if it is not necessary.'''
    global out_archivo
    if (not grabando)and (tiempo_archivo>tiempo_reinicio_video):
        if f'{nombre_archivo}.mp4' in os.listdir(f'{MODEL_DIR}/'):
            if 'out_archivo' in locals() or 'out_archivo' in globals():
                out_archivo.release()
                del(out_archivo)
            os.remove(f'{MODEL_DIR}/{nombre_archivo}.mp4')
        nombre_archivo = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        out_archivo = cv.VideoWriter(f'{OUTPUT_DIR}/{nombre_archivo}.mp4',fourcc,(1000/milisegundos),(frame_width,frame_height))
        return grabando,0,nombre_archivo,\
                1, time.time()
    elif grabando and (id_sospechoso is None) and indi==1:
        out_archivo.release()
        del(out_archivo)
        return False,tiempo_reinicio_video+0.1,None,\
                indi,t_inicio
       
    elif tiempo_archivo<=tiempo_reinicio_video:
        return grabando,time.time()-t_inicio,nombre_archivo,\
                indi,t_inicio
    

