# Imagen base.
FROM python:3.10.13

# Creamos usuario sin privilegios bajo el que corre la app para reducir brechas de seguridad.
# Ver https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#user
#ARG UID=10001 Éste no lo usaré
# RUN adduser \
#     --disabled-password \
#     --gecos "" \
#     --home "/nonexistent" \
#     --no-create-home \
#     keeper
#    --shell "/sbin/nologin" \ Lo comento porque sí quiero acceso interactivo a la shell
#    --uid "${UID}" \

# Variable de entorno para poder visualizar la cámara.
ENV DISPLAY $DISPLAY

# Carpeta donde irá el código de nuestra aplicación.
RUN mkdir -p /app/
# Establecemos /app/ como el directorio principal.
WORKDIR /app/
# Copiamos el proyecto en este directorio del contenedor.
COPY . .

# Instalamos requerimientos.
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

#Permisos de ejecución del archivo antes de crear el usuario.
# Dar permisos de ejecución al script
RUN chmod +x class_contagion/main.py

# Instalamos los paquetes necesarios para que corra el opencv
#RUN apt-get update && apt-get install -y libxcb-xinerama0 libxcb1 libxkbcommon-x11-0
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# # Cambiamos el propietario al nuevo usuario.
# RUN chown -R keeper:keeper .

# # Cambiar al nuevo usuario
# USER keeper

# Puerto que el container escucha.
EXPOSE 8001

# Comando que tiene que ejecutar para que la aplicación corra. ["comando", "argumentos"]
CMD ["python3", "class_contagion/main.py"]