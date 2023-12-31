# Verwenden Sie das offizielle Python-Image als Basis
FROM python:3.9-slim-buster

# Setzen Sie die Arbeitsverzeichnis im Container
WORKDIR /app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Kopieren Sie die Anwendungsabhängigkeiten (requirements.txt) in den Container
COPY requirements.txt requirements.txt

# install python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Kopieren Sie den gesamten Anwendungscode in den Container
COPY . .

# Starten Sie den Gunicorn-Server beim Start des Containers
CMD ["gunicorn", "--config", "gunicorn-cfg.py", "app:app"]
