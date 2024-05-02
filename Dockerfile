# Imagen base.
FROM python:3.10.13

# Variable de entorno para poder visualizar la cámara.
ENV DISPLAY $DISPLAY

# Carpeta donde irá el código de nuestra aplicación.
RUN mkdir -p /app/
# Establecemos /app/ como el directorio principal.
WORKDIR /app/

# Copiamos el proyecto en este directorio del contenedor.
COPY . /app/

# Instalamos requerimientos.
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Dar permisos de ejecución
#RUN chmod +x class_contagion/run.sh

EXPOSE 3100

# Instalamos los paquetes necesarios para que corra el opencv
RUN apt-get update && apt-get install -y libgl1-mesa-glx

#CMD ["bash", "class_contagion/run.sh"]
CMD ["python", "class_contagion/main.py"]