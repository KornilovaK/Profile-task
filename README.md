# Profile-task
ML-инженер

## Чтобы развернуть docker контейнер (docker dekstop уже установлен на пк):

* Через терминал клонируем репозиторий git clone https://github.com/KornilovaK/tg-style-bot.git
* Из основной директории переходим в docker cd docker
* Чтобы "построить" контейнер: docker compose -p <название нового контейнера> -f docker-compose.yml build
* Чтобы запустить его: docker compose -p <название нового контейнера> -f docker-compose.yml up

## Что содержит в себе app/:
* model.py Основной скрипт работы модели
* /models Содержит в себе сохраненные веса моделей
* /data Содержит в себе train_ds.csv, test_ds.csv, EDA.ipynb (предобработка данных), result.txt (результат работы модели - NDCG score для test выборки) и др файлы