FROM node:22-bookworm-slim

# Tesseract (OCR clásico, sin IA generativa) + idiomas + poppler (PDF)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      tesseract-ocr tesseract-ocr-spa tesseract-ocr-eng tesseract-ocr-por poppler-utils \
 && rm -rf /var/lib/apt/lists/*

ENV NODE_ENV=production \
    OMP_THREAD_LIMIT=1 \
    PORT=10000

WORKDIR /app
COPY server.js ./

USER node
EXPOSE 10000
CMD ["node", "server.js"]
