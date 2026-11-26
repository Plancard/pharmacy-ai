# 실행 이미지: 웹(Flask) + Agent(LangGraph·MCP) + CV(torch CPU · EasyOCR · YOLO)가 같은 앱 안에서 돈다 (명세서 2.1·2.5절).
# 모델 가중치(models/*.pt)는 저장소에 없다 → docker-compose.yml이 ./models를 읽기 전용으로 붙인다.
FROM python:3.12.16-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY web/requirements.txt web/requirements.txt
COPY agent/requirements.txt agent/requirements.txt
COPY cv/requirements.txt cv/requirements.txt
# ultralytics가 GUI판 opencv-python을 끌고 오는데, slim 이미지에는 GUI 라이브러리(libGL)가 없어 import가 실패한다.
# → GUI판을 지우고 headless판만 다시 깐다 (같은 cv2 폴더를 쓰므로 --force-reinstall)
RUN pip install -r web/requirements.txt -r agent/requirements.txt -r cv/requirements.txt \
    && pip uninstall -y opencv-python \
    && pip install --force-reinstall --no-deps opencv-python-headless==5.0.0.93

# 업로드 이미지(var/)를 호스트 폴더와 같은 사용자(UID 1000)로 쓰기 위해 root가 아닌 사용자로 돌린다
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/var \
    && chown app:app /app/var
USER app

# EasyOCR 모델(한국어·영어)을 빌드 때 받아 둔다: 시연장 네트워크가 막혀 있어도, 컨테이너를 새로 만들어도 다시 받지 않는다
RUN python -c "import easyocr; easyocr.Reader(['ko', 'en'], gpu=False, verbose=False)"

COPY --chown=app:app . .

EXPOSE 5000
# 스키마 적용(이미 있으면 건너뜀) → 재시작으로 끊긴 '처리 중' 처방 정리 → gunicorn 워커 1개 + 스레드 4개
CMD ["sh", "-c", "flask --app wsgi init-db && flask --app wsgi ensure-default-user && flask --app wsgi fail-stale --minutes 0 && exec gunicorn --workers 1 --threads 4 --bind 0.0.0.0:5000 --access-logfile - wsgi:app"]
