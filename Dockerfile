# 1차 실행 이미지: 웹(Flask) + 이후 CV·Agent가 같은 앱 안에서 돈다 (명세서 2.1·2.5절).
# CV(torch CPU)·Agent 의존성은 해당 모듈을 붙일 때 아래 RUN에 requirements를 추가한다.
FROM python:3.12.16-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY web/requirements.txt web/requirements.txt
RUN pip install -r web/requirements.txt

COPY . .

EXPOSE 5000
# 스키마 적용(이미 있으면 건너뜀) → gunicorn 워커 1개 + 스레드 4개
CMD ["sh", "-c", "flask --app wsgi init-db && exec gunicorn --workers 1 --threads 4 --bind 0.0.0.0:5000 --access-logfile - wsgi:app"]
