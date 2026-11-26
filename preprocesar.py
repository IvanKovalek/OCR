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


# ---------------------------------------------------------------- líneas y bordes
def mascara_lineas(g):
    """Líneas largas (continuas o punteadas), claras u oscuras, sin tocar las letras:
    1) solo trazos FINOS y ALARGADOS (guiones de un punteado, tramos de línea);
    2) entre esos, cadenas largas y alineadas = línea o borde."""
    h, w = g.shape
    total = np.zeros_like(g)
    grosor_max = max(4, min(14, min(h, w) // 150))
    largo = max(60, min(h, w) // 20)
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
        # b) líneas PUNTEADAS o cortadas: trazos sueltos finos y alargados, alineados en cadena
        n, et, st, _ = cv2.connectedComponentsWithStats(tinta, 8)
        ancho, alto = st[:, 2], st[:, 3]
        horiz = (ancho >= 1.8 * alto) & (alto <= grosor_max)
        vert = (alto >= 1.8 * ancho) & (ancho <= grosor_max)
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
        cerca = cv2.dilate(total, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31)))
        for tinta in (cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 41, 15),
                      cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, -15)):
            n, et, st, _ = cv2.connectedComponentsWithStats(cv2.bitwise_and(tinta, cv2.bitwise_not(total)), 8)
            for i in range(1, n):
                x, y, ww, hh, _ = st[i]
                if max(ww, hh) <= 60 and cerca[y:y + hh, x:x + ww].min() > 0:
                    total[et == i] = 255
    return cv2.dilate(total, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))


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
    for p in tesseract(collage, idioma_rescate, 11, 'r'):
        cy = p['y'] + p['h'] / 2
        for r, mt in zip(piezas, meta):
            if mt['oy'] <= cy < mt['oy'] + r.shape[0]:
                mt['palabras'].append(p)
                break

    # por recorte, la escala con mejor lectura (confianza media ponderada por caracteres)
    mejor, votos = {}, {}
    for mt in meta:
        ps = [p for p in mt['palabras'] if alfanum(p['texto']) > 0]
        if not ps:
            continue
        clave = ''.join(ch for p in ps for ch in p['texto'] if ch.isalnum()).lower()
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
        if votos[i].count(clave) < 2:
            continue
        for p in ps:
            # exigente: acá llegan restos de dibujos; un carácter suelto necesita mucha confianza
            minimo = 85 if alfanum(p['texto']) <= 1 else 70
            if p['conf'] < minimo:
                continue
            f = mt['f']
            salida.append({'linea': 'r:%d' % i,
                           'x': int(mt['ox'] + (p['x'] - 20) / f), 'y': int(mt['oy0'] + (p['y'] - mt['oy'] - 20) / f),
                           'w': int(p['w'] / f), 'h': int(p['h'] / f),
                           'conf': p['conf'], 'texto': p['texto']})
    return salida


# ---------------------------------------------------------------- principal
def main():
    idioma = sys.argv[1] if len(sys.argv) > 1 else 'spa'
    ruta = sys.argv[2] if len(sys.argv) > 2 else None
    g = cargar(ruta)
    g, f = escalar(g)

    lineas = mascara_lineas(g)
    if lineas.any():
        g = cv2.inpaint(g, lineas, 3, cv2.INPAINT_TELEA)

    normal = tesseract(g, idioma, 3, 'a')
    invertida = tesseract(255 - g, idioma, 3, 'b')
    elegidos = fusionar(normal, invertida)
    palabras = [p for l in elegidos for p in l['palabras']]
    palabras += rescatar(g, elegidos, idioma)

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
