FROM node:22-bookworm-slim

# Tesseract (OCR clásico, sin IA generativa) + idiomas + poppler (PDF)
# + Python con OpenCV para el preprocesado de imágenes (preprocesar.py)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      tesseract-ocr tesseract-ocr-spa tesseract-ocr-eng tesseract-ocr-por poppler-utils \
      python3 python3-pip \
 && pip3 install --no-cache-dir --break-system-packages numpy "opencv-python-headless>=4.8" \
 && apt-get purge -y python3-pip && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/* /root/.cache

ENV NODE_ENV=production \
    OMP_THREAD_LIMIT=1 \
    OPENCV_NUM_THREADS=1 \
    PORT=10000

WORKDIR /app
COPY server.js preprocesar.py ./

USER node
EXPOSE 10000
CMD ["node", "server.js"]
