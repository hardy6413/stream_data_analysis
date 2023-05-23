# Wybierz bazowy obraz
FROM python:3.9

# Skopiuj pliki aplikacji do kontenera
COPY . /app

# Ustaw katalog roboczy
WORKDIR /app

# Zainstaluj zależności
RUN pip install -r requirements.txt

# Uruchom aktualizację danych
CMD ["python", "main.py"]