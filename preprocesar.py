#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Preprocesado + OCR robusto (sin IA generativa: OpenCV clásico + Tesseract).
#
# Uso:  python3 preprocesar.py <idioma> [ruta]     (sin ruta: lee la imagen por stdin)
# Sale: JSON {"palabras": [{"linea","x","y","w","h","conf","texto"}...]}
#       en coordenadas de la imagen ORIGINAL. Node arma el texto final (columnas/párrafos).
#
# Pasos:
#  1. Aplana transparencias sobre blanco, pasa a grises y ajusta el tamaño
#     (agranda imágenes chicas: Tesseract lee mejor letras de ~30 px).
#  2. Borra líneas y bordes (tablas, recuadros, punteados) para que no se lean como letras.
#  3. Dos pasadas de Tesseract con umbral adaptativo (Sauvola, aguanta sombras y fondos
#     desparejos): texto oscuro sobre claro, y la imagen invertida (texto claro sobre oscuro).
#     Por cada zona se queda con el renglón más confiable.
#  4. Rescate: manchas de tinta que ninguna pasada leyó (números sueltos, palabras aisladas)
#     se recortan, se ponen derechas (texto oscuro), se juntan en una imagen chica y se leen
#     en una tercera pasada barata.
# -----------------------------------------------------------------------------
import sys, json, subprocess
import numpy as np
import cv2

LADO_MIN = 1600     # por debajo, se agranda (hasta x2)
LADO_MAX = 3200     # tope de tamaño de trabajo
MAX_RESCATES = 60   # recortes como máximo en el rescate
CONFUSIONES = str.maketrans({'b': '8', 'o': '0', 'l': '1', 'i': '1', '|': '1', 's': '5', 'z': '2', 'g': '9', 'q': '9'})


# ---------------------------------------------------------------- carga
def cargar(ruta):
    datos = open(ruta, 'rb').read() if ruta else sys.stdin.buffer.read()
    img = cv2.imdecode(np.frombuffer(datos, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError('No se pudo leer la imagen')
    if img.dtype != np.uint8:
        img = cv2.convertScaleAbs(img, alpha=255.0 / max(1, int(img.max())))
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        a = img[:, :, 3:4].astype(np.float32) / 255.0
        img = (img[:, :, :3].astype(np.float32) * a + 255.0 * (1 - a)).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def escalar(g):
    lado = max(g.shape)
    f = 1.0
    if lado < LADO_MIN:
        f = min(2.0, LADO_MAX / lado)
    elif lado > LADO_MAX:
        f = LADO_MAX / lado
    if f != 1.0:
        g = cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC if f > 1 else cv2.INTER_AREA)
    return g, f


# ---------------------------------------------------------------- enderezado
def angulo_inclinacion(g):
    """Ángulo (grados) que mejor alinea los renglones con la horizontal, entre -15 y 15.
    Usa el perfil de proyección sobre los bordes de la tinta (sirve para cualquier polaridad)."""
    f = min(1.0, 900.0 / max(g.shape))
    peq = cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else g
    grad = cv2.morphologyEx(peq, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    h, w = bw.shape
    # solo "letras": se descartan bordes, líneas y manchas grandes (dominarían el perfil en 0°)
    n, et, st, _ = cv2.connectedComponentsWithStats(bw, 8)
    grandes = (np.maximum(st[:, 2], st[:, 3]) > 0.12 * max(h, w)) | (st[:, 4] < 4)
    grandes[0] = True
    bw = np.where(grandes[et], 0, 255).astype(np.uint8)
    if bw.mean() < 0.2:
        return 0.0
    c = (w / 2, h / 2)

    def nitidez(a):
        r = cv2.warpAffine(bw, cv2.getRotationMatrix2D(c, a, 1), (w, h), flags=cv2.INTER_NEAREST)
        perfil = r.sum(axis=1, dtype=np.float64)
        return float(np.sum(np.diff(perfil) ** 2))

    mejor = max(np.arange(-15, 15.01, 1.0), key=nitidez)
    mejor = max(np.arange(mejor - 1, mejor + 1.01, 0.2), key=nitidez)
    return float(mejor)


def enderezar(g):
    a = angulo_inclinacion(g)
    if abs(a) < 2.9:     # Tesseract ya tolera inclinaciones chicas; rotar de más desordena
        return g
    h, w = g.shape
    # rotar sin recortar las esquinas
    m = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2; m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(g, m, (nw, nh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------- líneas y bordes
def mascara_lineas(g):
    """Líneas largas (continuas o punteadas), claras u oscuras, sin tocar las letras:
    1) solo trazos FINOS y ALARGADOS (guiones de un punteado, tramos de línea);
    2) entre esos, cadenas largas y alineadas = línea o borde."""
    h, w = g.shape
    total = np.zeros_like(g)
    rectas_h, rectas_v = [], []   # líneas continuas encontradas: (posición, desde, hasta)
    grosor_max = max(10, min(16, min(h, w) // 100))
    largo = max(80, min(h, w) // 20)
    fondo_claro = np.median(g) >= 128
    for tinta, es_oscura in (
        (cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 41, 15), True),
        (cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, -15), False),
    ):
        # a) líneas CONTINUAS (grillas de tablas, subrayados): tramos rectos largos y finos.
        #    Solo con la tinta que contrasta con el fondo general: en la otra polaridad, los
        #    espacios entre renglones parecen líneas y se borrarían letras.
        for horizontal in ((True, False) if es_oscura == fondo_claro else ()):
            # largo mínimo: 1/10 de la imagen (ningún trazo de letra es tan largo y fino)
            largo_r = max(largo, (w if horizontal else h) // 10)
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (largo_r, 1) if horizontal else (1, largo_r))
            rectas = cv2.morphologyEx(tinta, cv2.MORPH_OPEN, k)
            n2, et2, st2, _ = cv2.connectedComponentsWithStats(rectas, 8)
            for i in range(1, n2):
                x, y, ww, hh, _ = st2[i]
                grosor = hh if horizontal else ww
                if grosor > grosor_max * 1.5:
                    continue
                # una línea real tiene "vacío" de ambos lados; una franja entre texto, no
                m = 3
                if horizontal:
                    lados = [tinta[max(0, y - m - 5):max(0, y - m), x:x + ww], tinta[y + hh + m:y + hh + m + 5, x:x + ww]]
                else:
                    lados = [tinta[y:y + hh, max(0, x - m - 5):max(0, x - m)], tinta[y:y + hh, x + ww + m:x + ww + m + 5]]
                if max((l.mean() / 255.0) if l.size else 0 for l in lados) > 0.25:
                    continue
                total[et2 == i] = 255
                if horizontal:
                    rectas_h.append((y + hh // 2, x, x + ww))
                else:
                    rectas_v.append((x + ww // 2, y, y + hh))
        # b) líneas PUNTEADAS o cortadas: trazos sueltos finos y alargados, alineados en cadena
        n, et, st, _ = cv2.connectedComponentsWithStats(tinta, 8)
        ancho, alto = st[:, 2], st[:, 3]
        # guion = algo alargado pero corto (hasta ~4.5 veces su grosor); una "l" o un "|" es mucho más largo
        horiz = (ancho >= 1.8 * alto) & (ancho <= 4.5 * alto) & (alto <= grosor_max)
        vert = (alto >= 1.8 * ancho) & (alto <= 4.5 * ancho) & (ancho <= grosor_max)
        horiz[0] = vert[0] = False
        for sel, horizontal in ((horiz, True), (vert, False)):
            if not sel.any():
                continue
            trazos = np.where(sel[et], 255, 0).astype(np.uint8)
            k_cierre = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1) if horizontal else (1, 25))
            k_largo = cv2.getStructuringElement(cv2.MORPH_RECT, (largo, 1) if horizontal else (1, largo))
            cadenas = cv2.morphologyEx(cv2.morphologyEx(trazos, cv2.MORPH_CLOSE, k_cierre), cv2.MORPH_OPEN, k_largo)
            total |= cv2.bitwise_and(trazos, cv2.dilate(cadenas, np.ones((3, 3), np.uint8)))
    # restos chicos pegados a las líneas (esquinas redondeadas, puntas de un punteado)
    if total.any():
        cerca = cv2.dilate(total, cv2.getStructuringElement(cv2.MORPH_RECT, (45, 45)))
        for tinta in (cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 41, 15),
                      cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, -15)):
            n, et, st, _ = cv2.connectedComponentsWithStats(cv2.bitwise_and(tinta, cv2.bitwise_not(total)), 8)
            for i in range(1, n):
                x, y, ww, hh, _ = st[i]
                if max(ww, hh) <= 60 and (cerca[y:y + hh, x:x + ww] > 0).mean() > 0.6:
                    total[et == i] = 255
    return cv2.dilate(total, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))), rectas_h, rectas_v


def agrupar_pos(vals, tol=12):
    out = []
    for v in sorted(vals):
        if out and v - out[-1][-1] <= tol:
            out[-1].append(v)
        else:
            out.append([v])
    return [sum(g) / len(g) for g in out]


def detectar_tablas(rectas_h, rectas_v):
    """Grillas: al menos 3 líneas horizontales y 2 verticales que se cruzan.
    Devuelve [(caja, filas_y, columnas_x)]."""
    tablas = []
    usadas = set()
    for i, (yh, xh0, xh1) in enumerate(rectas_h):
        if i in usadas:
            continue
        hs = [j for j, (y, a, b) in enumerate(rectas_h) if min(b, xh1) - max(a, xh0) > 0.5 * min(b - a, xh1 - xh0)]
        if len(hs) < 3:
            continue
        y0 = min(rectas_h[j][0] for j in hs); y1 = max(rectas_h[j][0] for j in hs)
        x0 = min(rectas_h[j][1] for j in hs); x1 = max(rectas_h[j][2] for j in hs)
        vs = [v for v in rectas_v if x0 - 15 <= v[0] <= x1 + 15 and min(v[2], y1) - max(v[1], y0) > 0.5 * (y1 - y0)]
        cols_x = agrupar_pos([v[0] for v in vs])
        if len(cols_x) < 3:            # al menos 2 columnas reales; si no, no es una tabla
            continue
        usadas.update(hs)
        filas = agrupar_pos([rectas_h[j][0] for j in hs])
        cols = agrupar_pos([v[0] for v in vs])
        tablas.append(((x0, y0, x1, y1), filas, cols))
    return tablas


def marcar_tablas(palabras, tablas):
    """Cada palabra dentro de una grilla pasa a la "línea" de su fila, con su número de celda."""
    for p in palabras:
        cx, cy = p['x'] + p['w'] / 2, p['y'] + p['h'] / 2
        for t, ((x0, y0, x1, y1), filas, cols) in enumerate(tablas):
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                fila = sum(1 for f in filas if f < cy)
                col = sum(1 for c in cols if c < cx)
                p['linea'] = 't%d:%d' % (t, fila)
                p['celda'] = col
                break


# ---------------------------------------------------------------- Tesseract
def tesseract(img, idioma, psm, prefijo):
    ok, buf = cv2.imencode('.png', img)
    r = subprocess.run(['tesseract', 'stdin', 'stdout', '-l', idioma, '--psm', str(psm),
                        '-c', 'thresholding_method=2', 'tsv'],
                       input=buf.tobytes(), capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode('utf8', 'replace')[:300])
    palabras = []
    for fila in r.stdout.decode('utf8', 'replace').split('\n'):
        c = fila.split('\t')
        if len(c) < 12 or c[0] != '5':
            continue
        texto = ' '.join(c[11:]).strip()
        try:
            conf = float(c[10])
        except ValueError:
            continue
        if not texto or conf < 0:
            continue
        palabras.append({'linea': '%s:%s.%s.%s' % (prefijo, c[2], c[3], c[4]),
                         'x': int(c[6]), 'y': int(c[7]), 'w': int(c[8]), 'h': int(c[9]),
                         'conf': conf, 'texto': texto})
    return palabras


def alfanum(t):
    return sum(ch.isalnum() for ch in t)


def agrupar_lineas(palabras):
    lineas = {}
    for p in palabras:
        lineas.setdefault(p['linea'], []).append(p)
    out = []
    for ws in lineas.values():
        x0 = min(p['x'] for p in ws); y0 = min(p['y'] for p in ws)
        x1 = max(p['x'] + p['w'] for p in ws); y1 = max(p['y'] + p['h'] for p in ws)
        # puntaje: cuánto texto confiable aporta (la basura de baja confianza resta)
        puntaje = sum(alfanum(p['texto']) * (p['conf'] - 55) for p in ws)
        out.append({'caja': (x0, y0, x1, y1), 'palabras': ws, 'puntaje': puntaje})
    return out


def interseccion(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0]); h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0


def area(c):
    return max(1, (c[2] - c[0]) * (c[3] - c[1]))


def fusionar(*pasadas):
    """Por cada zona, el renglón con mejor puntaje; descarta los que se pisan con uno mejor."""
    candidatos = sorted((l for p in pasadas for l in agrupar_lineas(p)), key=lambda l: -l['puntaje'])
    elegidos = []
    for l in candidatos:
        if l['puntaje'] <= 0:
            continue
        if any(interseccion(l['caja'], e['caja']) > 0.3 * min(area(l['caja']), area(e['caja'])) for e in elegidos):
            continue
        # además, sacar palabras sueltas que ya están cubiertas por un renglón elegido (evita duplicados)
        tomadas = [(p['x'], p['y'], p['x'] + p['w'], p['y'] + p['h']) for e in elegidos for p in e['palabras']]
        libres = [p for p in l['palabras']
                  if not any(interseccion((p['x'], p['y'], p['x'] + p['w'], p['y'] + p['h']), t) > 0.3 * area((p['x'], p['y'], p['x'] + p['w'], p['y'] + p['h'])) for t in tomadas)]
        if not libres:
            continue
        if len(libres) < len(l['palabras']):
            l = dict(l, palabras=libres)
            if sum(alfanum(p['texto']) * (p['conf'] - 55) for p in libres) <= 0:
                continue
        elegidos.append(l)
    return elegidos


# ---------------------------------------------------------------- rescate
def manchas_de_texto(g):
    """Zonas con trazos (gradiente fuerte), agrupadas en palabras."""
    grad = cv2.morphologyEx(g, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5)))
    n, _, st, _ = cv2.connectedComponentsWithStats(bw, 8)
    h_img, w_img = g.shape
    cajas = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if h < 12 or w < 8 or h > h_img * 0.25 or w > w_img * 0.9:
            continue
        if a < 0.15 * w * h:          # contornos huecos grandes (marcos, círculos)
            continue
        cajas.append((x, y, x + w, y + h))
    return cajas


def rescatar(g, elegidos, idioma):
    ocupadas = [l['caja'] for l in elegidos]
    alturas = sorted(p['h'] for l in elegidos for p in l['palabras'] if alfanum(p['texto']) >= 3)
    alto_min = 0.45 * alturas[len(alturas) // 2] if alturas else 12
    pendientes = []
    for c in manchas_de_texto(g):
        if (c[3] - c[1]) < alto_min:       # mucho más chico que el texto del documento: ruido
            continue
        if any(interseccion(c, o) > 0.2 * area(c) for o in ocupadas):
            continue
        pendientes.append(c)
    pendientes = sorted(pendientes, key=lambda c: -area(c))[:MAX_RESCATES]
    # palabras dudosas de los renglones elegidos: se releen y se reemplazan si mejora
    dudosas = [p for l in elegidos for p in l['palabras'] if p['conf'] < 70 and alfanum(p['texto']) >= 2][:MAX_RESCATES]
    reemplazos = [None] * len(pendientes) + dudosas
    pendientes += [(p['x'], p['y'], p['x'] + p['w'], p['y'] + p['h']) for p in dudosas]
    if not pendientes:
        return []

    # Collage vertical: cada recorte derecho (texto oscuro sobre blanco) y a VARIAS escalas,
    # porque Tesseract es sensible al tamaño de letra: se queda la lectura más confiable.
    ESCALAS = (36, 48, 60)            # altura de texto objetivo, en px
    piezas, meta, y, ancho = [], [], 0, 0
    for i, (x0, y0, x1, y1) in enumerate(pendientes):
        m = max(6, (y1 - y0) // 3)
        base = g[max(0, y0 - m):y1 + m, max(0, x0 - m):x1 + m]
        _, bw = cv2.threshold(base, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        if (bw > 0).mean() < 0.5:     # la minoría clara es el texto -> invertir
            base = 255 - base
        for alto_obj in ESCALAS:
            f = min(4.0, max(0.3, alto_obj / max(1, (y1 - y0))))
            r = cv2.resize(base, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
            r = cv2.copyMakeBorder(r, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
            piezas.append(r)
            meta.append({'i': i, 'oy': y, 'f': f, 'ox': max(0, x0 - m), 'oy0': max(0, y0 - m), 'palabras': []})
            y += r.shape[0]; ancho = max(ancho, r.shape[1])
    collage = np.full((y, ancho), 255, np.uint8)
    for r, mt in zip(piezas, meta):
        collage[mt['oy']:mt['oy'] + r.shape[0], :r.shape[1]] = r

    # códigos, lotes y siglas se leen mucho mejor sumando el modelo en inglés
    idioma_rescate = idioma if 'eng' in idioma else idioma + '+eng'
    # dos modos: disperso (11) y bloque (6, cada recorte es un renglón). El 11 a veces
    # ignora un carácter suelto que el 6 lee bien; se elige por recorte más abajo.
    meta_total = []
    for psm in (11, 6):
        copia = [dict(mt, palabras=[], psm=psm) for mt in meta]
        for p in tesseract(collage, idioma_rescate, psm, 'r'):
            cy = p['y'] + p['h'] / 2
            for r, mt in zip(piezas, copia):
                if mt['oy'] <= cy < mt['oy'] + r.shape[0]:
                    mt['palabras'].append(p)
                    break
        meta_total += copia
    meta = meta_total

    # por recorte, la escala con mejor lectura (confianza media ponderada por caracteres)
    mejor, votos = {}, {}
    for mt in meta:
        ps = [p for p in mt['palabras'] if alfanum(p['texto']) > 0]
        if not ps or (reemplazos[mt['i']] is not None and mt['psm'] != 11):
            continue
        # para votar, los pares que Tesseract confunde cuentan como iguales (8/B, 0/O, 1/l/I...)
        clave = ''.join(ch for p in ps for ch in p['texto'] if ch.isalnum()).lower().translate(CONFUSIONES)
        votos.setdefault(mt['i'], []).append(clave)
        nota = sum(p['conf'] * alfanum(p['texto']) for p in ps) / sum(alfanum(p['texto']) for p in ps)
        if mt['i'] not in mejor or nota > mejor[mt['i']][0]:
            mejor[mt['i']] = (nota, mt, ps, clave)

    salida = []
    for i, (nota, mt, ps, clave) in mejor.items():
        if reemplazos[i] is not None:
            # palabra dudosa: se reemplaza solo por UNA palabra claramente más confiable
            original = reemplazos[i]
            if len(ps) == 1 and ps[0]['conf'] >= max(70, original['conf'] + 15) and alfanum(ps[0]['texto']) >= 2:
                original['texto'] = ps[0]['texto']
                original['conf'] = ps[0]['conf']
            continue
        # texto nuevo: tiene que leerse IGUAL en al menos dos escalas (la basura cambia con la escala)
        if votos[i].count(clave) < 2 and nota < 93:
            continue
        for p in ps:
            # exigente: acá llegan restos de dibujos; un carácter suelto necesita mucha confianza
            acuerdo = votos[i].count(clave) >= 2
            minimo = (75 if acuerdo else 85) if alfanum(p['texto']) <= 1 else 70
            if p['conf'] < minimo:
                continue
            f = mt['f']
            salida.append({'linea': 'r:%d' % i,
                           'x': int(mt['ox'] + (p['x'] - 20) / f), 'y': int(mt['oy0'] + (p['y'] - mt['oy'] - 20) / f),
                           'w': int(p['w'] / f), 'h': int(p['h'] / f),
                           'conf': p['conf'], 'texto': p['texto']})
    return salida


# ---------------------------------------------------------------- imágenes difíciles
def nota_lectura(elegidos):
    """Confianza media ponderada por caracteres de lo leído."""
    ps = [p for l in elegidos for p in l['palabras'] if alfanum(p['texto']) > 0]
    n = sum(alfanum(p['texto']) for p in ps)
    return (sum(p['conf'] * alfanum(p['texto']) for p in ps) / n if n else 0.0), n


def cobertura(g, elegidos):
    """Qué parte de las manchas con aspecto de texto quedó cubierta por lo leído."""
    manchas = manchas_de_texto(g)
    if not manchas:
        return 1.0
    cajas = [l['caja'] for l in elegidos]
    cubiertas = sum(1 for c in manchas if any(interseccion(c, o) > 0.3 * area(c) for o in cajas))
    return cubiertas / len(manchas)


def wiener_movimiento(g, largo, vertical=False, k=0.006):
    """Deshace un desenfoque de movimiento recto (deconvolución de Wiener)."""
    pad = largo * 2
    a = cv2.copyMakeBorder(g, pad, pad, pad, pad, cv2.BORDER_REFLECT).astype(np.float32) / 255
    psf = np.zeros((1, largo), np.float32) + 1.0 / largo
    if vertical:
        psf = psf.T
    H = np.fft.fft2(psf, s=a.shape)
    r = np.real(np.fft.ifft2(np.conj(H) / (np.abs(H) ** 2 + k) * np.fft.fft2(a)))
    r = np.roll(r, (-(psf.shape[0] // 2), -(psf.shape[1] // 2)), (0, 1))[pad:-pad, pad:-pad]
    return np.clip(r * 255, 0, 255).astype(np.uint8)


def largo_desenfoque(g, vertical=False, max_largo=90):
    """Largo de un desenfoque de movimiento: la autocorrelación de la derivada
    tiene un valle negativo justo en ese largo."""
    a = g.astype(np.float32)
    if vertical:
        a = a.T
    d = np.diff(a, axis=1)
    d = d - d.mean(axis=1, keepdims=True)
    F = np.fft.rfft(d, n=2 * d.shape[1], axis=1)
    ac = np.fft.irfft(F * np.conj(F), axis=1)[:, :max_largo].mean(axis=0)
    ac = ac / (ac[0] + 1e-6)
    return int(np.argmin(ac[3:]) + 3)


def direccion_desenfoque(g):
    """<0.75: movida en horizontal; >1.33: movida en vertical; si no, ~1."""
    gx = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0)).mean()
    gy = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1)).mean()
    return gx / (gy + 1e-6)


def tamano_bloque(g):
    """Si la imagen está pixelada (agrandada sin suavizar), el lado del bloque; si no, 0."""
    d = np.abs(np.diff(g.astype(np.int16), axis=1)) > 8
    cols = np.nonzero(d.any(axis=0))[0] + 1
    if len(cols) < 20:
        return 0
    mejor = 0
    for b in range(3, 17):
        frac = np.bincount(cols % b, minlength=b).max() / len(cols)
        if frac > 0.85:
            mejor = b
            break
    return mejor


def variantes_dificiles(g, f, g_crudo):
    """Versiones mejoradas de una imagen difícil: (nombre, imagen). f = escala de trabajo.
    g_crudo = la imagen antes de borrar líneas (para deshacer el movimiento)."""
    k_med = max(3, int(round(4.5 * f)) | 1)          # mediana proporcional a la escala
    out = [('nitida', cv2.addWeighted(g, 2.5, cv2.GaussianBlur(g, (0, 0), 1.5 * f), -1.5, 0)),
           ('sin_ruido', cv2.medianBlur(g, k_med))]
    r = direccion_desenfoque(g_crudo)
    if r < 0.75 or r > 1.33:
        vertical = r > 1.33
        L = largo_desenfoque(g_crudo, vertical)
        for k in (0.008, 0.02):
            out.append(('movida%d' % int(k * 1000), wiener_movimiento(g_crudo, L, vertical, k)))
    return out


def despixelar(g):
    """Imagen agrandada sin suavizar (bloques): se reduce al tamaño real y se agranda suave."""
    b = tamano_bloque(g)
    if not b:
        return g
    h, w = g.shape
    peq = cv2.resize(g, (max(1, w // b), max(1, h // b)), interpolation=cv2.INTER_AREA)
    return cv2.resize(peq, (w, h), interpolation=cv2.INTER_CUBIC)


def pasadas(img, idioma, prefijo, escala=1.0, invertir=None):
    """Pasadas de Tesseract sobre img (normal y/o invertida), con coordenadas llevadas a la escala base."""
    res = []
    for inv in ((False, True) if invertir is None else (invertir,)):
        ps = tesseract(255 - img if inv else img, idioma, 3, prefijo + ('i' if inv else 'n'))
        if escala != 1.0:
            for p in ps:
                for k in ('x', 'y', 'w', 'h'):
                    p[k] = int(p[k] / escala)
        res.append(ps)
    return res


# ---------------------------------------------------------------- tachado
def tachada(g, p, palabras):
    """¿La palabra está tachada? Una línea fina a media altura, más larga que ~2 letras,
    que la cruza. (El travesaño de una 'e' o un '4' es corto; un subrayado está abajo.)"""
    if alfanum(p['texto']) == 0:
        return False
    hermanas = [o for o in palabras if o['linea'] == p['linea'] and alfanum(o['texto']) > 0]
    h = sorted(o['h'] for o in hermanas)[len(hermanas) // 2]
    x0 = max(0, p['x'] - h); x1 = p['x'] + p['w'] + h
    recorte = g[max(0, p['y']):p['y'] + p['h'], x0:x1]
    if recorte.size == 0 or recorte.shape[0] < 6:
        return False
    # tinta = la clase minoritaria del recorte (sirve para texto claro u oscuro)
    _, bw = cv2.threshold(recorte, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if (bw > 0).mean() > 0.5:
        bw = 255 - bw
    hh = recorte.shape[0]
    banda = bw[int(0.3 * hh):int(0.75 * hh), :]
    largo = max(8, int(1.3 * h))
    if banda.shape[1] < largo:
        return False
    trazos = cv2.morphologyEx(banda, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (largo, 1)))
    filas = np.nonzero(trazos.any(axis=1))[0]
    if len(filas) == 0 or len(filas) > 0.35 * banda.shape[0]:   # una franja gruesa no es una tachadura
        return False
    cols = np.nonzero(trazos.any(axis=0))[0] + x0
    dentro = np.sum((cols >= p['x']) & (cols < p['x'] + p['w']))
    return dentro >= 0.5 * p['w']


# ---------------------------------------------------------------- principal
def main():
    idioma = sys.argv[1] if len(sys.argv) > 1 else 'spa'
    ruta = sys.argv[2] if len(sys.argv) > 2 else None
    g = cargar(ruta)
    g = despixelar(g)
    g, f = escalar(g)
    g = enderezar(g)   # (las coordenadas quedan en el marco enderezado: solo se usan para el orden)

    g_crudo = g        # sin borrar líneas: en una foto movida, las letras estiradas parecen líneas
    lineas, rectas_h, rectas_v = mascara_lineas(g)
    tablas = detectar_tablas(rectas_h, rectas_v)
    if lineas.any():
        g = cv2.inpaint(g, lineas, 3, cv2.INPAINT_TELEA)

    candidatos = pasadas(g, idioma, 'a')
    elegidos = fusionar(*candidatos)

    # Letras enormes: Tesseract lee mejor letras de ~50 px
    alturas = sorted(p['h'] for l in elegidos for p in l['palabras'] if alfanum(p['texto']) >= 2)
    if alturas and alturas[len(alturas) // 2] > 160:
        fs = 50.0 / alturas[len(alturas) // 2]
        chica = cv2.resize(g, None, fx=fs, fy=fs, interpolation=cv2.INTER_AREA)
        candidatos += pasadas(chica, idioma, 'z', fs)
        elegidos = fusionar(*candidatos)

    # Imagen difícil (borrosa, movida, con ruido, pixelada): variantes mejoradas
    nota, _ = nota_lectura(elegidos)
    if nota < 85 or cobertura(g, elegidos) < 0.9:
        claros = sum(alfanum(p['texto']) for l in elegidos for p in l['palabras'] if p['linea'].startswith('ai'))
        oscuros = sum(alfanum(p['texto']) for l in elegidos for p in l['palabras'] if p['linea'].startswith('an'))
        invertir = claros > oscuros       # polaridad dominante: una sola pasada por variante
        for i, (nombre, v) in enumerate(variantes_dificiles(g, f, g_crudo)):
            candidatos += pasadas(v, idioma, 'v%d' % i, invertir=invertir)
        elegidos = fusionar(*candidatos)
        # un renglón que SOLO aparece en una variante tiene que ser muy confiable (evita basura de dibujos)
        cajas_base = [l['caja'] for c in candidatos[:2] for l in agrupar_lineas(c) if l['puntaje'] > 0]
        _, chars_base = nota_lectura(fusionar(*candidatos[:2]))
        minimo = 80 if chars_base >= 10 else 55     # si la base no leyó casi nada, se confía más en las variantes
        elegidos = [l for l in elegidos if not l['palabras'][0]['linea'].startswith('v')
                    or any(interseccion(l['caja'], b) > 0.3 * area(l['caja']) for b in cajas_base)
                    or nota_lectura([l])[0] >= minimo]

    palabras = [p for l in elegidos for p in l['palabras']]
    # lo rescatado se suma al renglón que tiene al lado (si lo hay), para no partir el renglón
    for r in rescatar(g, elegidos, idioma):
        for l in elegidos:
            x0, y0, x1, y1 = l['caja']
            solape = min(y1, r['y'] + r['h']) - max(y0, r['y'])
            hueco = max(x0 - (r['x'] + r['w']), r['x'] - x1)
            if solape > 0.5 * min(r['h'], y1 - y0) and hueco < 3 * r['h']:
                r['linea'] = l['palabras'][0]['linea']
                break
        else:
            # sin renglón al lado: se agrupa con otras palabras rescatadas de la misma fila
            for o in palabras:
                if o['linea'].startswith('r') and o['linea'] != r['linea']:
                    solape = min(o['y'] + o['h'], r['y'] + r['h']) - max(o['y'], r['y'])
                    if solape > 0.5 * min(o['h'], r['h']):
                        r['linea'] = o['linea']
                        break
        palabras.append(r)
    # tachaduras (en una foto movida todo parece "tachado": ahí no se busca)
    if 0.75 <= direccion_desenfoque(g_crudo) <= 1.33:
        palabras = [p for p in palabras if not tachada(g, p, palabras)]
    if tablas:
        marcar_tablas(palabras, tablas)

    for p in palabras:  # volver a coordenadas de la imagen original
        for k in ('x', 'y', 'w', 'h'):
            p[k] = round(p[k] / f)
    sys.stdout.write(json.dumps({'palabras': palabras}, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        sys.stderr.write(str(e))
        sys.exit(2)
