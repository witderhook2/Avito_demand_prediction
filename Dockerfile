FROM python:3.11-slim

WORKDIR /app

#отключаем буферизацию вывода 
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

#устанавливаем минимально необходимые системные пакеты (libgomp1 нужен для CatBoost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app_api.py"]
