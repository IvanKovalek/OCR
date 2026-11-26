// =============================================================================
// OCR — Servidor Node para Render (plan gratis, deploy con Docker).
// Página web en "/" + endpoint POST /api/ocr. OCR con Tesseract nativo (sin IA generativa).
// Sin dependencias de npm. Los archivos se procesan en memoria y se descartan
// (los PDF usan una carpeta temporal que se borra apenas termina el request).
// =============================================================================

const http = require('node:http');
const { spawn } = require('node:child_process');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');

const PORT = Number(process.env.PORT || 10000);
const MAX_BYTES = 10 * 1024 * 1024;                              // 10 MB por archivo
const MAX_ARCHIVOS = Number(process.env.MAX_ARCHIVOS || 10);     // por request (API)
const MAX_TOTAL = Number(process.env.MAX_TOTAL_MB || 30) * 1024 * 1024; // por request
const MAX_PAGINAS = Number(process.env.MAX_PAGINAS || 30);       // PDF escaneados
const CONCURRENCIA = Number(process.env.CONCURRENCIA || 1);      // trabajos OCR simultáneos
const MAX_COLA = Number(process.env.MAX_COLA || 100);
const TIMEOUT_MS = Number(process.env.TIMEOUT_MS || 300000);     // por imagen/página
const RATE_LIMIT = Number(process.env.RATE_LIMIT || 120);        // requests/min por IP (0 = sin límite)
const ALLOWED_ORIGIN = process.env.ALLOWED_ORIGIN || '*';
const IDIOMAS = ['spa', 'eng', 'por', 'spa+eng'];

class HttpError extends Error {
  constructor(status, codigo, mensaje, detalles) {
    super(mensaje); this.status = status; this.codigo = codigo; this.detalles = detalles;
  }
}

// -----------------------------------------------------------------------------
// Cola: limita trabajos simultáneos para no pasarse de los 512 MB de RAM
// -----------------------------------------------------------------------------
let activos = 0;
const cola = [];
function enCola(fn) {
  if (cola.length >= MAX_COLA) {
    return Promise.reject(new HttpError(503, 'SERVIDOR_OCUPADO', 'El servidor está ocupado.', 'Probá de nuevo en unos segundos.'));
  }
  return new Promise((resolve, reject) => {
    cola.push({ fn, resolve, reject });
    siguiente();
  });
}
function siguiente() {
  if (activos >= CONCURRENCIA || cola.length === 0) return;
  const { fn, resolve, reject } = cola.shift();
  activos++;
  Promise.resolve().then(fn).then(resolve, reject).finally(() => { activos--; siguiente(); });
}

// -----------------------------------------------------------------------------
// Ejecutar binarios (tesseract, pdftotext, pdftoppm, pdfinfo)
// -----------------------------------------------------------------------------
function ejecutar(cmd, args, entrada) {
  return new Promise((resolve, reject) => {
    const p = spawn(cmd, args, { env: { ...process.env, OMP_THREAD_LIMIT: '1' } });
    const out = [], err = [];
    const timer = setTimeout(() => { p.kill('SIGKILL'); }, TIMEOUT_MS);
    p.stdout.on('data', (c) => out.push(c));
    p.stderr.on('data', (c) => err.push(c));
    p.on('error', (e) => { clearTimeout(timer); reject(e); });
    p.on('close', (code, signal) => {
      clearTimeout(timer);
      if (signal === 'SIGKILL') return reject(new HttpError(504, 'TIEMPO_AGOTADO', 'El OCR tardó demasiado.', 'Probá con una imagen más chica o más nítida.'));
      if (code !== 0) return reject(new Error(cmd + ' salió con código ' + code + ': ' + Buffer.concat(err).toString().slice(0, 300)));
      resolve(Buffer.concat(out).toString('utf8'));
    });
    p.stdin.on('error', () => {}); // si el proceso cierra stdin antes de tiempo
    if (entrada) p.stdin.end(entrada); else p.stdin.end();
  });
}

function limpiar(texto) {
  return texto.replace(/\f/g, '').replace(/[ \t]+\n/g, '\n').replace(/\n{3,}/g, '\n\n').trim();
}

// -----------------------------------------------------------------------------
// Detección de formato por "magic bytes"
// -----------------------------------------------------------------------------
function detectarTipo(b) {
  if (b.length < 12) return 'desconocido';
  if (b[0] === 0xff && b[1] === 0xd8 && b[2] === 0xff) return 'jpeg';
  if (b[0] === 0x89 && b[1] === 0x50 && b[2] === 0x4e && b[3] === 0x47) return 'png';
  if (b[0] === 0x25 && b[1] === 0x50 && b[2] === 0x44 && b[3] === 0x46) return 'pdf';
  if (b.toString('ascii', 0, 4) === 'RIFF' && b.toString('ascii', 8, 12) === 'WEBP') return 'webp';
  if (b.toString('ascii', 0, 3) === 'GIF') return 'gif';
  if (b[0] === 0x42 && b[1] === 0x4d) return 'bmp';
  if ((b[0] === 0x49 && b[1] === 0x49 && b[2] === 0x2a) || (b[0] === 0x4d && b[1] === 0x4d && b[3] === 0x2a)) return 'tiff';
  if (b.toString('ascii', 4, 8) === 'ftyp') return 'heic';
  return 'desconocido';
}

// -----------------------------------------------------------------------------
// OCR de un archivo
// -----------------------------------------------------------------------------
// Tesseract en modo TSV: devuelve cada palabra con su posición y confianza.
// "entrada" puede ser un Buffer (se pasa por stdin, sin tocar disco) o una ruta.
async function tesseractPalabras(entrada, idioma) {
  const esRuta = typeof entrada === 'string';
  const tsv = await ejecutar('tesseract',
    [esRuta ? entrada : 'stdin', 'stdout', '-l', idioma, '--psm', '3', 'tsv'],
    esRuta ? null : entrada);
  const palabras = [];
  for (const fila of tsv.split('\n')) {
    const c = fila.split('\t');
    if (c.length < 12 || c[0] !== '5') continue; // nivel 5 = palabra
    const texto = c.slice(11).join(' ').trim();
    const conf = Number(c[10]);
    if (!texto || !(conf >= 0)) continue;
    palabras.push({ linea: c[2] + '.' + c[3] + '.' + c[4], x: +c[6], y: +c[7], w: +c[8], h: +c[9], conf, texto });
  }
  return palabras;
}

const mediana = (arr) => { const s = arr.slice().sort((a, b) => a - b); return s[s.length >> 1] || 0; };

function armarSegmento(ws) {
  const x = Math.min(...ws.map((w) => w.x));
  const y = Math.min(...ws.map((w) => w.y));
  return {
    x, y,
    der: Math.max(...ws.map((w) => w.x + w.w)),
    abajo: Math.max(...ws.map((w) => w.y + w.h)),
    alto: mediana(ws.map((w) => w.h)),
    conf: ws.reduce((s, w) => s + w.conf, 0) / ws.length,
    palabras: ws,
    texto: ws.map((w) => w.texto).join(' '),
  };
}

const solapeX = (a, b) => Math.min(a.der, b.der) - Math.max(a.x, b.x);
const solapeY = (a, b) => Math.min(a.abajo, b.abajo) - Math.max(a.y, b.y);

// Reconstruye el orden de lectura a partir de las posiciones de las palabras:
//  1) corta renglones donde hay huecos grandes (columnas, tarjetas lado a lado)
//  2) descarta ruido (logos e íconos leídos como letras sueltas con baja confianza)
//  3) agrupa los trozos en "flujos" (columnas / párrafos) y los ordena
function reconstruirTexto(palabras) {
  // 0) Ruido a nivel palabra: íconos/logos que Tesseract lee como 1-2 caracteres con baja confianza
  palabras = palabras.filter((w) => {
    const utiles = w.texto.replace(/[^\p{L}\p{N}]/gu, '').length;
    if (utiles === 0 && w.conf < 60) return false;
    if (utiles === 0 && w.texto.length === 1 && !'-–—·•&%=/:'.includes(w.texto)) return false;
    if (utiles <= 2 && w.conf < 40) return false;
    return w.conf >= 15;
  });

  // 1) Renglones de Tesseract -> segmentos
  const lineas = new Map();
  for (const w of palabras) {
    if (!lineas.has(w.linea)) lineas.set(w.linea, []);
    lineas.get(w.linea).push(w);
  }
  let segmentos = [];
  for (let ws of lineas.values()) {
    // Íconos/logos: "letra" suelta (1-2 caracteres) mucho más alta que las palabras del renglón
    const hTexto = mediana(ws.filter((w) => w.texto.replace(/[^\p{L}\p{N}]/gu, '').length >= 3).map((w) => w.h));
    if (hTexto) {
      // ...o que no está apoyada en el renglón (su base queda muy arriba o muy abajo de la de las palabras)
      const base = mediana(ws.filter((w) => w.texto.replace(/[^\p{L}\p{N}]/gu, '').length >= 3).map((w) => w.y + w.h));
      ws = ws.filter((w) => {
        if (w.texto.replace(/[^\p{L}\p{N}]/gu, '').length > 2 || w.conf >= 90) return true;
        if (w.h > 1.6 * hTexto) return false;
        return Math.abs(w.y + w.h - base) <= 0.45 * hTexto;
      });
    }
    if (ws.length === 0) continue;
    ws.sort((a, b) => a.x - b.x);
    const umbral = Math.max(2.5 * mediana(ws.map((w) => w.h)), 24);
    let actual = [ws[0]];
    for (let i = 1; i < ws.length; i++) {
      const hueco = ws[i].x - (ws[i - 1].x + ws[i - 1].w);
      if (hueco > umbral) { segmentos.push(armarSegmento(actual)); actual = []; }
      actual.push(ws[i]);
    }
    segmentos.push(armarSegmento(actual));
  }

  // 2) Ruido: trozos cortos con baja confianza, o cualquier trozo casi ilegible
  segmentos = segmentos.filter((s) => {
    const utiles = s.texto.replace(/[^\p{L}\p{N}]/gu, '').length;
    if (utiles === 0) return false;
    if (s.conf < 50 && utiles <= 3) return false;
    return s.conf >= 30;
  });
  if (segmentos.length === 0) return { texto: '', confianza: 0 };

  // 3) Flujos: un trozo continúa un flujo si está justo debajo y alineado con él
  segmentos.sort((a, b) => a.y - b.y || a.x - b.x);
  const flujos = [];
  for (const s of segmentos) {
    const vecinos = segmentos.filter((o) => o !== s && solapeY(o, s) > 0.5 * Math.min(o.abajo - o.y, s.abajo - s.y));
    let mejor = null, mejorHueco = Infinity;
    for (const f of flujos) {
      const u = f.ultimo;
      const alto = Math.max(u.alto, s.alto);
      const hueco = s.y - u.abajo;
      if (hueco < -0.5 * alto || hueco > 1.3 * alto) continue;              // muy lejos verticalmente
      if (Math.max(u.alto, s.alto) > 2 * Math.min(u.alto, s.alto)) continue; // tamaño de letra muy distinto (logo/título)
      if (solapeX(u, s) < 0.3 * Math.min(u.der - u.x, s.der - s.x)) continue; // no alineado
      if (vecinos.some((v) => solapeX(u, v) > 0)) continue;                 // u abarca varias columnas
      if (hueco < mejorHueco) { mejor = f; mejorHueco = hueco; }
    }
    if (mejor) { mejor.segs.push(s); mejor.ultimo = s; }
    else flujos.push({ segs: [s], ultimo: s, y: s.y, x: s.x, alto: s.alto });
  }

  // Orden de lectura: de arriba hacia abajo; si arrancan a la misma altura, de izquierda a derecha
  flujos.sort((a, b) => {
    const tolerancia = 0.6 * Math.max(a.alto, b.alto);
    return Math.abs(a.y - b.y) <= tolerancia ? a.x - b.x : a.y - b.y;
  });

  // Una letra suelta y aislada (no un número) casi siempre es un dibujo: flecha, ícono, viñeta.
  // Se conserva si hay otro texto en la misma fila (ej. opciones "A  B  C" o celdas de tabla).
  for (let i = flujos.length - 1; i >= 0; i--) {
    const f = flujos[i];
    if (f.segs.length !== 1 || f.segs[0].palabras.length !== 1) continue;
    if (!/^\p{L}[.,;:!?¡¿)]?$/u.test(f.segs[0].palabras[0].texto)) continue;
    const s0 = f.segs[0];
    const enFila = segmentos.some((o) => o !== s0 && solapeY(o, s0) > 0.5 * Math.min(o.abajo - o.y, s0.abajo - s0.y));
    if (!enFila) flujos.splice(i, 1);
  }

  // Flujos de un solo trozo alineados en la misma fila (números en recuadros, celdas sueltas)
  // se escriben en una sola línea separados por espacios, en vez de un bloque cada uno.
  const bloques = [];
  for (const f of flujos) {
    const prev = bloques[bloques.length - 1];
    if (f.segs.length === 1 && prev && prev.fila &&
        Math.abs(prev.y - f.y) <= 0.6 * Math.max(prev.alto, f.alto) && f.x > prev.der) {
      prev.texto += '   ' + f.segs[0].texto;
      prev.der = f.segs[0].der;
      continue;
    }
    bloques.push({
      texto: f.segs.map((s) => s.texto).join('\n'),
      fila: f.segs.length === 1, y: f.y, alto: f.alto, der: f.segs[0].der,
    });
  }

  const usadas = segmentos.flatMap((s) => s.palabras);
  return {
    texto: bloques.map((b) => b.texto).join('\n\n'),
    confianza: Math.round(usadas.reduce((t, w) => t + w.conf, 0) / usadas.length) / 100,
  };
}

async function ocrImagen(entrada, idioma, modo) {
  if (modo === 'crudo') {
    const esRuta = typeof entrada === 'string';
    const t = await ejecutar('tesseract', [esRuta ? entrada : 'stdin', 'stdout', '-l', idioma, '--psm', '3'], esRuta ? null : entrada);
    return { texto: limpiar(t) };
  }
  return reconstruirTexto(await palabrasRobustas(entrada, idioma));
}

// Preprocesado (OpenCV) + varias pasadas de Tesseract: ver preprocesar.py.
// Si Python fallara por algo inesperado, se cae a una pasada simple de Tesseract.
const SCRIPT_PREPROCESO = path.join(__dirname, 'preprocesar.py');
async function palabrasRobustas(entrada, idioma) {
  const esRuta = typeof entrada === 'string';
  try {
    const salida = await ejecutar('python3', [SCRIPT_PREPROCESO, idioma].concat(esRuta ? [entrada] : []), esRuta ? null : entrada);
    return JSON.parse(salida).palabras;
  } catch (e) {
    if (e instanceof HttpError) throw e;
    console.error('Preprocesado falló, uso Tesseract directo:', e.message);
    return tesseractPalabras(entrada, idioma);
  }
}

async function ocrPdf(buffer, idioma, modo) {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'ocr-'));
  try {
    const archivo = path.join(dir, 'doc.pdf');
    await fs.writeFile(archivo, buffer);

    let paginas = 0, info = '';
    try {
      info = await ejecutar('pdfinfo', [archivo]);
      const m = info.match(/^Pages:\s+(\d+)/m);
      paginas = m ? Number(m[1]) : 0;
    } catch (e) {
      throw new HttpError(422, 'PDF_INVALIDO', 'No se pudo abrir el PDF (¿está protegido o dañado?).');
    }

    // 1) PDF digital: extraer la capa de texto (instantáneo)
    // (sin -layout: así respeta el orden de lectura de las columnas)
    const texto = limpiar(await ejecutar('pdftotext', ['-enc', 'UTF-8', archivo, '-']));
    if (texto.replace(/\s+/g, '').length >= 20) {
      return { tipo: 'pdf', metodo: 'capa-de-texto', paginas, texto };
    }

    // 2) PDF escaneado: rasterizar cada página y pasarla por OCR
    const total = Math.min(paginas || 1, MAX_PAGINAS);
    // Resolución según el tamaño de la página: ~3000 px de lado mayor (entre 72 y 300 dpi)
    const tam = info.match(/^Page size:\s+([\d.]+) x ([\d.]+) pts/m);
    const ladoPts = tam ? Math.max(Number(tam[1]), Number(tam[2])) : 842;
    const dpi = Math.round(Math.min(300, Math.max(72, 3000 / (ladoPts / 72))));
    await ejecutar('pdftoppm', ['-r', String(dpi), '-gray', '-png', '-f', '1', '-l', String(total), archivo, path.join(dir, 'p')]);
    const imagenes = (await fs.readdir(dir)).filter((f) => f.startsWith('p') && f.endsWith('.png')).sort();
    const partes = [], confs = [];
    for (let i = 0; i < imagenes.length; i++) {
      const r = await ocrImagen(path.join(dir, imagenes[i]), idioma, modo);
      partes.push((imagenes.length > 1 ? '--- Página ' + (i + 1) + ' ---\n' : '') + r.texto);
      if (r.confianza) confs.push(r.confianza);
      await fs.rm(path.join(dir, imagenes[i]), { force: true });
    }
    let final = partes.join('\n\n');
    if (paginas > MAX_PAGINAS) final += '\n\n[Se procesaron solo las primeras ' + MAX_PAGINAS + ' de ' + paginas + ' páginas]';
    const res = { tipo: 'pdf', metodo: 'ocr', paginas: total, texto: final };
    if (confs.length) res.confianza = Math.round((confs.reduce((a, b) => a + b, 0) / confs.length) * 100) / 100;
    return res;
  } finally {
    await fs.rm(dir, { recursive: true, force: true });
  }
}

async function procesarArchivo(buffer, nombre, idioma, modo) {
  if (!buffer || buffer.length === 0) throw new HttpError(400, 'SIN_ARCHIVO', 'El archivo está vacío.');
  if (buffer.length > MAX_BYTES) throw new HttpError(413, 'ARCHIVO_DEMASIADO_GRANDE', 'El archivo supera los 10 MB.');
  const tipo = detectarTipo(buffer);
  if (tipo === 'desconocido' || tipo === 'heic') {
    throw new HttpError(415, 'FORMATO_NO_SOPORTADO',
      'Formato no soportado' + (tipo === 'heic' ? ' (HEIC)' : '') + '.',
      'Acepta JPG, PNG, WEBP, GIF, BMP, TIFF y PDF. La página web convierte otros formatos automáticamente.');
  }
  return enCola(async () => {
    if (tipo === 'pdf') return { nombre, ...(await ocrPdf(buffer, idioma, modo)) };
    return { nombre, tipo: 'imagen', formato: tipo, metodo: 'ocr', ...(await ocrImagen(buffer, idioma, modo)) };
  });
}

// -----------------------------------------------------------------------------
// HTTP: utilidades
// -----------------------------------------------------------------------------
const CORS = {
  'Access-Control-Allow-Origin': ALLOWED_ORIGIN,
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
};

function enviarJson(res, status, data) {
  res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store', ...CORS });
  res.end(JSON.stringify(data));
}

// Lee el cuerpo como stream con tope de tamaño (corta antes de llenar la RAM)
function cuerpoLimitado(req, max) {
  let total = 0;
  return new ReadableStream({
    start(ctrl) {
      req.on('data', (c) => {
        total += c.length;
        if (total > max) {
          ctrl.error(new HttpError(413, 'ARCHIVO_DEMASIADO_GRANDE', 'La petición supera el tamaño máximo.', 'Máximo 10 MB por archivo.'));
          req.pause();
          return;
        }
        ctrl.enqueue(new Uint8Array(c.buffer, c.byteOffset, c.length));
      });
      req.on('end', () => { try { ctrl.close(); } catch (_) {} });
      req.on('error', (e) => { try { ctrl.error(e); } catch (_) {} });
    },
  });
}

const rateLimitMap = new Map();
function excedeLimite(ip) {
  if (!RATE_LIMIT) return false;
  const now = Date.now();
  if (rateLimitMap.size > 5000) for (const [k, v] of rateLimitMap) if (now > v.reset) rateLimitMap.delete(k);
  const d = rateLimitMap.get(ip);
  if (!d || now > d.reset) { rateLimitMap.set(ip, { n: 1, reset: now + 60000 }); return false; }
  if (d.n >= RATE_LIMIT) return true;
  d.n++; return false;
}

function errorJson(e, nombre) {
  if (e instanceof HttpError) return { ok: false, nombre, error: e.message, codigo: e.codigo, detalles: e.detalles };
  console.error('Error:', e);
  return { ok: false, nombre, error: 'Error interno del servidor.', detalles: String(e && e.message) };
}

// -----------------------------------------------------------------------------
// Endpoint /api/ocr
// -----------------------------------------------------------------------------
async function manejarOcr(req, res, url) {
  const ip = String(req.headers['x-forwarded-for'] || req.socket.remoteAddress || '').split(',')[0].trim();
  if (excedeLimite(ip)) {
    return enviarJson(res, 429, { ok: false, error: 'Demasiadas peticiones', detalles: 'Límite: ' + RATE_LIMIT + ' por minuto.' });
  }

  const ct = String(req.headers['content-type'] || '');
  let idioma = url.searchParams.get('idioma') || 'spa';
  // modo=auto (por defecto: reconstruye columnas y párrafos) | crudo (salida tal cual de Tesseract)
  let modo = url.searchParams.get('modo') || 'auto';
  let archivos = [];

  try {
    const declarado = Number(req.headers['content-length'] || 0);
    if (ct.includes('multipart/form-data')) {
      if (declarado > MAX_TOTAL) throw new HttpError(413, 'ARCHIVO_DEMASIADO_GRANDE', 'La petición supera ' + (MAX_TOTAL / 1048576) + ' MB.', 'Mandá menos archivos por petición.');
      // Node 22 trae un parser de multipart estándar (Request.formData)
      const peticion = new Request('http://local/', { method: 'POST', headers: { 'content-type': ct }, body: cuerpoLimitado(req, MAX_TOTAL), duplex: 'half' });
      let form;
      try { form = await peticion.formData(); }
      catch (e) { if (e instanceof HttpError) throw e; if (e && e.cause instanceof HttpError) throw e.cause; throw new HttpError(400, 'MULTIPART_INVALIDO', 'No se pudo leer el formulario.', String(e && e.message)); }
      idioma = String(form.get('idioma') || idioma);
      modo = String(form.get('modo') || modo);
      for (const campo of ['archivos', 'archivo', 'file', 'files']) {
        for (const f of form.getAll(campo)) {
          if (f && typeof f !== 'string') archivos.push({ nombre: f.name || 'archivo', tam: f.size, blob: f });
        }
      }
    } else if (ct.startsWith('image/') || ct.startsWith('application/pdf') || ct.startsWith('application/octet-stream')) {
      if (declarado > MAX_BYTES) throw new HttpError(413, 'ARCHIVO_DEMASIADO_GRANDE', 'El archivo supera los 10 MB.');
      const buf = Buffer.from(await new Response(cuerpoLimitado(req, MAX_BYTES)).arrayBuffer());
      archivos.push({ nombre: url.searchParams.get('nombre') || 'archivo', tam: buf.length, buffer: buf });
    } else {
      throw new HttpError(400, 'SIN_ARCHIVO', "Mandá un multipart con el campo 'archivo' o el binario con Content-Type image/* o application/pdf.");
    }

    if (archivos.length === 0) throw new HttpError(400, 'SIN_ARCHIVO', "Falta el campo 'archivo'.");
    if (archivos.length > MAX_ARCHIVOS) throw new HttpError(413, 'DEMASIADOS_ARCHIVOS', 'Máximo ' + MAX_ARCHIVOS + ' archivos por petición.');
    if (!IDIOMAS.includes(idioma)) throw new HttpError(400, 'IDIOMA_INVALIDO', 'Idioma no soportado: ' + idioma, 'Usá: ' + IDIOMAS.join(', '));
  } catch (e) {
    req.resume();
    const j = errorJson(e);
    return enviarJson(res, e instanceof HttpError ? e.status : 500, j);
  }

  // Procesar (en orden; la cola limita la concurrencia global)
  const resultados = [];
  for (const a of archivos) {
    try {
      if (a.tam > MAX_BYTES) throw new HttpError(413, 'ARCHIVO_DEMASIADO_GRANDE', 'El archivo supera los 10 MB.');
      const buffer = a.buffer || Buffer.from(await a.blob.arrayBuffer());
      a.blob = null; a.buffer = null;
      resultados.push({ ok: true, ...(await procesarArchivo(buffer, a.nombre, idioma, modo === 'crudo' ? 'crudo' : 'auto')) });
    } catch (e) {
      resultados.push({ ...errorJson(e, a.nombre), _status: e instanceof HttpError ? e.status : 500 });
    }
  }

  if (resultados.length === 1) {
    const r = resultados[0];
    const status = r.ok ? 200 : r._status;
    delete r._status;
    return enviarJson(res, status, { idioma, ...r });
  }
  resultados.forEach((r) => delete r._status);
  return enviarJson(res, 200, { ok: true, idioma, cantidad: resultados.length, resultados });
}

// -----------------------------------------------------------------------------
// Servidor
// -----------------------------------------------------------------------------
const inicio = new Date().toISOString();
let ultimoPing = null;

const servidor = http.createServer(async (req, res) => {
  const url = new URL(req.url, 'http://localhost');
  try {
    const lectura = req.method === 'GET' || req.method === 'HEAD';
    if (req.method === 'OPTIONS') { res.writeHead(204, CORS); return res.end(); }

    // --- KEEP-ALIVE para UptimeRobot (o cualquier monitor) ---
    // Configurá un monitor HTTP(s) a https://TU-APP.onrender.com/ping cada 5 minutos.
    // Responde al instante (GET o HEAD), no pasa por la cola ni por el rate limit.
    if (lectura && url.pathname === '/ping') {
      ultimoPing = new Date().toISOString();
      res.writeHead(200, { 'Content-Type': 'text/plain; charset=utf-8', 'Cache-Control': 'no-store' });
      return res.end('ok');
    }

    if (lectura && url.pathname === '/') {
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8', ...CORS });
      return res.end(HTML_PAGE);
    }
    if (lectura && url.pathname === '/api/salud') {
      return enviarJson(res, 200, { ok: true, cola: cola.length, activos, ultimoPing, encendidoDesde: inicio, maxBytes: MAX_BYTES, maxArchivos: MAX_ARCHIVOS, idiomas: IDIOMAS });
    }
    if (req.method === 'POST' && url.pathname === '/api/ocr') return await manejarOcr(req, res, url);
    res.writeHead(404, { 'Content-Type': 'text/plain', ...CORS });
    res.end('Not Found');
  } catch (e) {
    if (!res.headersSent) enviarJson(res, 500, errorJson(e));
    else res.end();
  }
});
servidor.requestTimeout = 0; // los PDF escaneados pueden tardar varios minutos
servidor.listen(PORT, () => console.log('OCR escuchando en el puerto ' + PORT));

// -----------------------------------------------------------------------------
// Página web (sin comillas invertidas ni "${" dentro del HTML)
// -----------------------------------------------------------------------------
const HTML_PAGE = String.raw`<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OCR - Extraer texto de fotos y documentos</title>
<style>
  :root { --azul:#1d4ed8; --azul-claro:#e8efff; --borde:#d8dde6; --texto:#1f2937; --suave:#6b7280; --ok:#15803d; --err:#b91c1c; }
  * { box-sizing: border-box; }
  body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background:#f4f6fa; color:var(--texto); margin:0; padding:24px 16px; }
  main { max-width: 860px; margin: auto; }
  h1 { margin: 0 0 4px; font-size: 1.6rem; }
  .sub { color: var(--suave); margin: 0 0 20px; }
  .card { background:#fff; border-radius:12px; box-shadow:0 2px 12px rgba(0,0,0,.06); padding:20px; margin-bottom:16px; }
  .zona { border:2px dashed var(--borde); border-radius:10px; padding:32px 16px; text-align:center; cursor:pointer; transition:.15s; }
  .zona:hover, .zona.arrastrando { border-color: var(--azul); background: var(--azul-claro); }
  .zona strong { color: var(--azul); }
  .zona small { display:block; color:var(--suave); margin-top:6px; }
  .fila { display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin-top:14px; }
  select, button { font: inherit; padding: 9px 14px; border-radius: 8px; border:1px solid var(--borde); background:#fff; }
  button { cursor:pointer; }
  button.prim { background: var(--azul); border-color: var(--azul); color:#fff; }
  button:disabled { opacity:.5; cursor:not-allowed; }
  .estado { margin-top:14px; min-height:1.4em; color:var(--suave); }
  .estado.err { color: var(--err); } .estado.ok { color: var(--ok); }
  .barra { height:6px; background:#eef1f6; border-radius:3px; overflow:hidden; margin-top:8px; display:none; }
  .barra > div { height:100%; width:0; background:var(--azul); transition: width .2s; }
  .item { border:1px solid var(--borde); border-radius:10px; padding:12px; margin-top:12px; }
  .item-cab { display:flex; justify-content:space-between; gap:10px; align-items:center; flex-wrap:wrap; }
  .item-cab b { word-break: break-all; }
  .meta { color: var(--suave); font-size: .9rem; }
  .item.err { border-color:#f3c4c4; background:#fff7f7; }
  .item.err .meta { color: var(--err); }
  textarea { width:100%; min-height:180px; margin-top:10px; padding:12px; border-radius:8px; border:1px solid var(--borde); font: 14px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; resize: vertical; }
  .peq { padding:5px 10px; font-size:.85rem; }
  input[type=file] { display:none; }
  code { background:#f1f3f7; padding:2px 6px; border-radius:4px; }
</style>
</head>
<body>
<main>
  <h1>OCR</h1>
  <p class="sub">Extraé el texto de fotos, capturas, documentos escaneados y PDF. Podés subir varios archivos a la vez. No se guarda nada.</p>

  <div class="card">
    <div class="zona" id="zona" tabindex="0">
      <strong>Elegí archivos</strong> o arrastralos acá (también podés pegar con Ctrl+V)
      <small>Fotos (JPG, PNG, WEBP, GIF, BMP, HEIC en Safari…) y PDF · máximo 10 MB cada uno</small>
    </div>
    <input type="file" id="archivo" accept="image/*,application/pdf" multiple>
    <input type="file" id="camara" accept="image/*" capture="environment">
    <div class="fila">
      <label for="idioma">Idioma del texto:</label>
      <select id="idioma">
        <option value="spa" selected>Español</option>
        <option value="eng">Inglés</option>
        <option value="por">Portugués</option>
        <option value="spa+eng">Español + Inglés</option>
      </select>
      <button id="btn-camara" type="button">📷 Sacar foto</button>
    </div>
    <div class="barra" id="barra"><div></div></div>
    <div class="estado" id="estado"></div>
  </div>

  <div class="card" id="card-resultados" style="display:none">
    <div class="fila" style="margin-top:0; justify-content:space-between">
      <strong>Resultados</strong>
      <span>
        <button class="peq" id="btn-copiar-todo" type="button">Copiar todo</button>
        <button class="peq" id="btn-descargar-todo" type="button">Descargar todo (.txt)</button>
        <button class="peq" id="btn-limpiar" type="button">Limpiar</button>
      </span>
    </div>
    <div id="lista"></div>
  </div>

  <p class="meta">API: <code>POST /api/ocr</code> con el campo <code>archivo</code> (multipart) · imágenes y PDF · máx. 10 MB por archivo.</p>
</main>

<script>
(function () {
  var MAX_BYTES = 10 * 1024 * 1024;
  var MAX_LADO = 2500;   // las fotos se reducen antes de enviarse (mucho más rápido)
  var $ = function (id) { return document.getElementById(id); };
  var estado = $('estado'), barra = $('barra'), lista = $('lista'), cardRes = $('card-resultados');
  var pendientes = [], trabajando = false, hechos = 0, totalLote = 0;
  var resultados = [];

  function setEstado(t, tipo) { estado.textContent = t || ''; estado.className = 'estado' + (tipo ? ' ' + tipo : ''); }
  function setProgreso(p) {
    if (p == null) { barra.style.display = 'none'; return; }
    barra.style.display = 'block'; barra.firstElementChild.style.width = Math.round(p * 100) + '%';
  }

  $('zona').addEventListener('click', function () { $('archivo').click(); });
  $('zona').addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') $('archivo').click(); });
  $('btn-camara').addEventListener('click', function () { $('camara').click(); });
  $('archivo').addEventListener('change', function () { agregar(this.files); this.value = ''; });
  $('camara').addEventListener('change', function () { agregar(this.files); this.value = ''; });
  ['dragenter', 'dragover'].forEach(function (ev) { $('zona').addEventListener(ev, function (e) { e.preventDefault(); $('zona').classList.add('arrastrando'); }); });
  ['dragleave', 'drop'].forEach(function (ev) { $('zona').addEventListener(ev, function (e) { e.preventDefault(); $('zona').classList.remove('arrastrando'); }); });
  $('zona').addEventListener('drop', function (e) { agregar(e.dataTransfer.files); });
  document.addEventListener('paste', function (e) {
    var items = (e.clipboardData || {}).items || [], fs = [];
    for (var i = 0; i < items.length; i++) if (items[i].kind === 'file') { var f = items[i].getAsFile(); if (f) fs.push(f); }
    if (fs.length) agregar(fs);
  });

  function textoTotal() {
    return resultados.filter(function (r) { return r.ok; }).map(function (r) {
      return '===== ' + r.nombre + ' =====\n' + r.area.value;
    }).join('\n\n');
  }
  $('btn-copiar-todo').addEventListener('click', function () {
    navigator.clipboard.writeText(textoTotal()).then(function () { setEstado('Texto copiado.', 'ok'); });
  });
  $('btn-descargar-todo').addEventListener('click', function () { descargar(textoTotal(), 'ocr-resultados.txt'); });
  $('btn-limpiar').addEventListener('click', function () { if (trabajando) return; lista.innerHTML = ''; resultados = []; cardRes.style.display = 'none'; setEstado(''); });

  function descargar(texto, nombre) {
    var a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([texto], { type: 'text/plain;charset=utf-8' }));
    a.download = nombre; a.click();
    setTimeout(function () { URL.revokeObjectURL(a.href); }, 1000);
  }

  function crearItem(nombre) {
    var div = document.createElement('div'); div.className = 'item';
    var cab = document.createElement('div'); cab.className = 'item-cab';
    var b = document.createElement('b'); b.textContent = nombre;
    var meta = document.createElement('span'); meta.className = 'meta'; meta.textContent = 'En espera…';
    cab.appendChild(b); cab.appendChild(meta); div.appendChild(cab);
    lista.appendChild(div); cardRes.style.display = 'block';
    return { div: div, meta: meta, nombre: nombre };
  }

  function agregar(files) {
    for (var i = 0; i < files.length; i++) {
      var f = files[i], item = crearItem(f.name || 'archivo');
      if (f.size > MAX_BYTES) { marcarError(item, 'Pesa ' + (f.size / 1048576).toFixed(1) + ' MB (máximo 10 MB).'); continue; }
      pendientes.push({ file: f, item: item });
      totalLote++;
    }
    if (!trabajando) procesarCola();
  }

  function marcarError(item, msg) {
    item.div.className = 'item err'; item.meta.textContent = msg;
    resultados.push({ ok: false, nombre: item.nombre });
  }

  function marcarOk(item, data, seg) {
    var area = document.createElement('textarea'); area.spellcheck = false; area.value = data.texto || '';
    var fila = document.createElement('div'); fila.className = 'fila';
    var bc = document.createElement('button'); bc.className = 'peq prim'; bc.textContent = 'Copiar';
    bc.onclick = function () { navigator.clipboard.writeText(area.value).then(function () { setEstado('Copiado: ' + item.nombre, 'ok'); }); };
    var bd = document.createElement('button'); bd.className = 'peq'; bd.textContent = 'Descargar .txt';
    bd.onclick = function () { descargar(area.value, item.nombre.replace(/\.[^.]+$/, '') + '.txt'); };
    fila.appendChild(bc); fila.appendChild(bd);
    item.div.appendChild(area); item.div.appendChild(fila);
    var partes = [];
    if (data.paginas) partes.push(data.paginas + (data.paginas === 1 ? ' página' : ' páginas'));
    partes.push(data.metodo === 'capa-de-texto' ? 'texto del PDF' : 'OCR');
    partes.push(seg + ' s');
    if (!data.texto) partes.push('sin texto detectado');
    item.meta.textContent = partes.join(' · ');
    resultados.push({ ok: true, nombre: item.nombre, area: area });
  }

  // Fotos: se convierten a JPEG reducido con fondo blanco (soporta cualquier formato que abra el navegador)
  function prepararImagen(file) {
    var carga = window.createImageBitmap ? createImageBitmap(file).catch(function () { return conImg(file); }) : conImg(file);
    return carga.then(function (img) {
      var w = img.naturalWidth || img.width, h = img.naturalHeight || img.height;
      // PNG/JPG que ya tienen buen tamaño: se mandan tal cual (sin recomprimir = texto más nítido)
      if ((file.type === 'image/png' || file.type === 'image/jpeg') && Math.max(w, h) <= MAX_LADO) return file;
      var esc = Math.min(1, MAX_LADO / Math.max(w, h));
      var c = document.createElement('canvas');
      c.width = Math.round(w * esc); c.height = Math.round(h * esc);
      var ctx = c.getContext('2d');
      ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, c.width, c.height);
      ctx.drawImage(img, 0, 0, c.width, c.height);
      return new Promise(function (ok) { c.toBlob(ok, 'image/jpeg', 0.92); });
    });
  }
  function conImg(file) {
    return new Promise(function (ok, mal) {
      var img = new Image();
      img.onload = function () { ok(img); };
      img.onerror = function () { mal(new Error('Tu navegador no puede abrir este formato. Convertilo a JPG o PNG.')); };
      img.src = URL.createObjectURL(file);
    });
  }

  function enviar(blob, nombre) {
    var fd = new FormData();
    fd.append('archivo', blob, nombre);
    fd.append('idioma', $('idioma').value);
    return fetch('/api/ocr', { method: 'POST', body: fd }).then(function (r) {
      return r.json().catch(function () { return { ok: false, error: 'Respuesta inválida del servidor (HTTP ' + r.status + ').' }; });
    });
  }

  function procesarCola() {
    var t = pendientes.shift();
    if (!t) {
      trabajando = false; setProgreso(null);
      setEstado('Listo: ' + hechos + ' de ' + totalLote + ' archivo(s) procesados.', 'ok');
      hechos = 0; totalLote = 0;
      return;
    }
    trabajando = true;
    setProgreso(hechos / totalLote);
    setEstado('Procesando ' + (hechos + 1) + ' de ' + totalLote + ': ' + t.item.nombre + ' (si el servidor estaba dormido, el primero puede tardar ~1 minuto)');
    t.item.meta.textContent = 'Procesando…';
    var t0 = performance.now();
    var esPdf = t.file.type === 'application/pdf' || /\.pdf$/i.test(t.file.name || '');
    var paso = esPdf
      ? Promise.resolve(t.file)
      : prepararImagen(t.file);
    paso.then(function (blob) {
      var nombreEnvio = (esPdf || blob === t.file) ? (t.file.name || 'archivo') : (t.file.name || 'imagen').replace(/\.[^.]+$/, '') + '.jpg';
      return enviar(blob, nombreEnvio);
    }).then(function (data) {
      if (!data.ok) throw new Error(data.detalles ? data.error + ' ' + data.detalles : data.error);
      marcarOk(t.item, data, ((performance.now() - t0) / 1000).toFixed(1));
    }).catch(function (e) {
      marcarError(t.item, e && e.message ? e.message : 'Problema de conexión.');
    }).then(function () {
      hechos++; procesarCola();
    });
  }
})();
</script>
</body>
</html>`;
