"""Genera 11_vencer_xgboost.ipynb (sección nueva del proyecto). Se ejecuta luego con construir_cuaderno.py."""
import os
import nbformat as nbf

DIR = os.path.dirname(os.path.abspath(__file__))
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

celdas = [
    md("""# 11. Intento de superar a XGBoost con algoritmos originales (Top 30)

Este capítulo ejecuta **los 30 algoritmos** del documento *Top 30 de algoritmos originales para intentar vencer a
XGBoost con significancia*: primero la ruta recomendada (#1–#4, #6, #7, #11, #12, #26) y después una ampliación
prerregistrada con los 21 restantes. Cada candidato se programó con la API de NumPy (en GPU, con CuPy como sustituto directo de
NumPy) y se comparó contra un XGBoost ajustado con **el mismo presupuesto** y **las mismas variables**.

Estructura:

1. El documento de partida (Top 30).
2. Protocolo prerregistrado.
3. Datos e igualdad de condiciones.
4. El motor propio de boosting.
5. XGBoost de referencia.
6. Compuerta de cribado de cada variante.
7. Protocolo completo y Super Learner.
8. Pruebas de hipótesis y veredicto.
9. Conclusiones y reproducción."""),
    code("""import json, os, glob
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Markdown, display
from sklearn.metrics import roc_auc_score
RES = 'resultados'
leer = lambda n: json.load(open(os.path.join(RES, n), encoding='utf-8'))
pd.set_option('display.float_format', lambda x: f'{x:.5f}')
TOP30 = {  # nombre en el código -> (# del documento, nombre, probabilidad a priori del documento)
    'sl': (1, 'Super Learner (Caruana)', 'Media (la mejor)'), 'sl2': (1, 'Super Learner + réplicas (#1 + #11)', 'Media'),
    'sl3': (1, 'Super Learner, biblioteca ampliada', 'Media'), 'hojas_lineales': (2, 'GBDT con hojas lineales', 'Media'),
    'heterogeneo': (3, 'Newton heterogéneo', 'Media'), 'newton': (4, 'Newton propio', 'Media-baja (requisito)'),
    'ktboost': (5, 'Árbol + núcleo (KTBoost)', 'Media-baja'), 'olvidadizo': (6, 'Árboles olvidadizos', 'Media-baja'),
    'dart': (7, 'DART', 'Media-baja'), 'rotaciones': (8, 'Rotaciones aleatorias', 'Media-baja'),
    'oblicuos': (9, 'Cortes oblicuos dispersos', 'Media-baja'), 'rgf': (10, 'RGF totalmente correctivo', 'Media-baja'),
    'newton_s1': (11, 'Réplica por semilla (MultiBoosting)', 'Baja'), 'mvs': (12, 'MVS', 'Baja'),
    'infiniteboost': (13, 'InfiniteBoost', 'Baja'), 'lce': (14, 'Cascada local (LCE)', 'Baja'),
    'bstacking': (15, 'bstacking (metamodelo logístico)', 'Baja-media'), 'rankboost': (16, 'RankBoost (pares, AUC)', 'Baja'),
    'trboost': (17, 'TRBoost', 'Baja'), 'nesterov': (18, 'Nesterov', 'Baja'), 'focal': (19, 'Pérdida focal', 'Baja'),
    'bosque_profundo': (20, 'Bosque profundo en cascada', 'Baja'), 'bosque_rotacion': (21, 'Bosque de rotación', 'Baja'),
    'extra_trees': (22, 'Árboles extremadamente aleatorios', 'Baja'), 'spe': (23, 'Self-paced ensemble', 'Baja'),
    'des': (24, 'Selección dinámica', 'Baja'), 'lmt': (25, 'Árbol de modelo logístico', 'Muy baja'),
    'goss': (26, 'GOSS', 'Muy baja en AUC'), 'bart': (27, 'BART probit', 'Muy baja'),
    'psplines': (28, 'Boosting con P-splines', 'Muy baja'), 'ga2m': (29, 'GA²M/EBM', 'Muy baja'),
    'ngboost': (30, 'NGBoost', 'Muy baja'),
}"""),
    md("## 11.1 Documento de partida"),
    code("display(Markdown(open('TOP30_algoritmos.md', encoding='utf-8').read()))"),
    md("""## 11.2 Protocolo prerregistrado

Las reglas se fijaron **antes** de correr los candidatos y no se cambiaron después (la segunda ronda se
escribió antes de conocer el resultado del Super Learner). El archivo completo:"""),
    code("display(Markdown(open('PROTOCOLO.md', encoding='utf-8').read()))"),
    md("""## 11.3 Datos e igualdad de condiciones

`features.py` construye una sola matriz a partir de las siete tablas de Home Credit (solicitud + agregados de
buró, solicitudes previas, POS, cuotas y tarjeta, con indicadores de presencia de historial). XGBoost y todos los
candidatos reciben exactamente esa matriz."""),
    code("""cols = json.load(open('cache/columnas.json'))
y = np.load('cache/y.npy')
P = np.load('cache/particiones.npz')
print(f'Filas: {len(y):,} | variables: {len(cols)} | tasa de incumplimiento: {y.mean():.4f}')
print({k: len(P[k]) for k in P.files})
pd.Series([c.split('_')[0] for c in cols]).value_counts().head(12).rename('variables por prefijo').to_frame()"""),
    md("""## 11.4 Motor propio de boosting (`gbdt.py`)

- Boosting de Newton (segundo orden) sobre la pérdida logística, con histogramas de 255 cubetas + una cubeta para
  faltantes, dirección aprendida para los faltantes, crecimiento por niveles y truco de la resta de histogramas.
- Regularización como en Chen y Guestrin (2016): λ (L2), α (L1, umbral suave), γ (ganancia mínima), hessiano
  mínimo por hoja, submuestreo de filas y de columnas, peso de la clase positiva.
- Variantes del Top 30 activables por parámetro: #2 hojas lineales, #3 profundidad heterogénea, #6 árboles
  olvidadizos, #7 DART, #12 MVS, #26 GOSS, #29 un árbol por variable, #11 réplicas por semilla, #1 Super Learner
  (`superlearner.py`, selección de Caruana con pesos cruzados por pliegue).
- Ampliación (`modelos.py`): #5 KTBoost, #8 rotaciones, #9 cortes oblicuos, #10 RGF, #13 InfiniteBoost, #14 LCE,
  #15 bstacking, #16 RankBoost, #17 TRBoost, #18 Nesterov, #19 pérdida focal, #20 bosque profundo, #21 bosque de
  rotación, #22 árboles extremadamente aleatorios, #23 SPE, #24 selección dinámica, #25 LMT, #27 BART probit,
  #28 P-splines, #29 GA²M y #30 NGBoost.
- No usa xgboost, lightgbm, catboost, torch, tensorflow ni `sklearn.ensemble`."""),
    code("print(open('gbdt.py', encoding='utf-8').read())"),
    code("print(open('modelos.py', encoding='utf-8').read())"),
    md("## 11.5 XGBoost de referencia (GPU, TPE de 20 evaluaciones)"),
    code("""x = leer('xgb_ajuste.json')
display(pd.Series(x['params']).to_frame('hiperparámetro'))
print(f"AUC diseño {x['auc_diseno']:.5f} | AUC disjunta {x['auc_disjunta']:.5f} | rondas {x['it']}")"""),
    md("""## 11.6 Compuerta de cribado

Cada variante recibió el mismo TPE de 20 evaluaciones (mismo espacio y semilla del muestreador que XGBoost) en la
partición de diseño, y su mejor configuración se evaluó en la partición disjunta. Para pasar debe superar a
XGBoost en las dos."""),
    code("""filas = []
for p in sorted(glob.glob(os.path.join(RES, 'compuerta_*.json'))):
    c = json.load(open(p, encoding='utf-8'))
    num, nom, prob = TOP30[c['variante']]
    filas.append(dict(n=num, algoritmo=nom, prob_a_priori=prob, variante=c['variante'], auc_diseno=c['auc_diseno'], dif_diseno=c['auc_diseno'] - x['auc_diseno'],
                      auc_disjunta=c['auc_disjunta'], dif_disjunta=c['auc_disjunta'] - x['auc_disjunta'],
                      pasa=c['pasa_compuerta'], minutos=c['segundos'] / 60))
comp = pd.DataFrame(filas).sort_values('auc_diseno', ascending=False)
print(f'{len(comp)} algoritmos pasaron por la compuerta; la pasan: {int(comp.pasa.sum())}')
display(comp)
fig, ax = plt.subplots(figsize=(12, 4.5))
w = 0.38; i = np.arange(len(comp))
ax.bar(i - w/2, comp['dif_diseno'] * 1000, w, label='diseño')
ax.bar(i + w/2, comp['dif_disjunta'] * 1000, w, label='disjunta')
ax.axhline(0, color='k', lw=0.8)
ax.set_xticks(i, [f"#{a} {b}" for a, b in zip(comp['n'], comp['variante'])], rotation=60, ha='right', fontsize=8)
ax.set_ylabel('AUC variante − AUC XGBoost (×1000)')
ax.set_title('Compuerta de cribado'); ax.legend(); plt.tight_layout(); plt.show()"""),
    md("""## 11.7 Protocolo completo y Super Learner

Validación cruzada estratificada 5 × 2 sobre el 80 % de entrenamiento (246,005 filas) y prueba en el 20 % restante
(61,502 filas). El Super Learner combina márgenes estandarizados de los boostings propios con pesos de Caruana que,
en cada pliegue, se estiman sin usar ese pliegue."""),
    code("""_, ytr, yte = None, y[P['tr']], y[P['te']]
filas = []
for p in sorted(glob.glob(os.path.join(RES, 'pred_*.npz'))):
    d = np.load(p); n = os.path.basename(p)[5:-4]
    if np.isnan(d['auc_pliegues']).any():
        continue
    filas.append(dict(modelo=n, auc_oof=roc_auc_score(ytr, d['oof0']), auc_prueba=roc_auc_score(yte, d['test']),
                      media_pliegues=d['auc_pliegues'].mean(), de_pliegues=d['auc_pliegues'].std(ddof=1)))
tabla = pd.DataFrame(filas).sort_values('auc_oof', ascending=False).reset_index(drop=True)
display(tabla)
for n in ('sl', 'sl2', 'sl3', 'bstacking'):
    if os.path.exists(os.path.join(RES, f'superlearner_{n}.json')):
        s = leer(f'superlearner_{n}.json')
        print(n, 'pesos (prueba):', {k: round(v, 3) for k, v in s['pesos_test'].items() if abs(v) > 1e-9})"""),
    code("""base = np.load(os.path.join(RES, 'pred_xgb.npz'))['auc_pliegues']
fig, ax = plt.subplots(figsize=(9, 4))
for n in tabla['modelo']:
    if n == 'xgb':
        continue
    d = np.load(os.path.join(RES, f'pred_{n}.npz'))['auc_pliegues'] - base
    ax.plot(range(1, 11), d * 1000, marker='o', label=n, lw=2 if n.startswith('sl') else 0.8)
ax.axhline(0, color='k', lw=0.8)
ax.set_xlabel('pliegue (rep. 0: 1–5, rep. 1: 6–10)'); ax.set_ylabel('AUC − AUC XGBoost (×1000)')
ax.legend(fontsize=7, ncol=3); ax.set_title('Diferencia por pliegue frente a XGBoost'); plt.tight_layout(); plt.show()"""),
    md("""## 11.8 Pruebas de hipótesis y veredicto

DeLong sobre las predicciones fuera de pliegue, DeLong sobre la prueba, t corregida de Nadeau–Bengio y Wilcoxon
sobre las 10 diferencias por pliegue; p bilaterales con corrección de Holm sobre los candidatos que llegaron al
protocolo completo. «SUPERA» exige diferencia positiva y las cuatro p ajustadas < 0.05."""),
    code("""R = pd.DataFrame(leer('pruebas_final.json'))
cols_v = ['candidato', 'auc_oof', 'auc_oof_xgb', 'dif_oof', 'auc_test', 'auc_test_xgb', 'dif_test', 'pliegues_ganados',
          'p_delong_oof_holm', 'p_delong_test_holm', 'p_nb_holm', 'p_wilcoxon_holm', 'veredicto']
display(R[cols_v])"""),
    md("## 11.9 Conclusiones"),
    code("""gan = R[R['veredicto'] == 'SUPERA']
if len(gan):
    g = gan.sort_values('dif_oof', ascending=False).iloc[0]
    texto = (f"**Hay ganador: `{g['candidato']}`.** Supera a XGBoost con AUC fuera de pliegue {g['auc_oof']:.5f} frente a "
             f"{g['auc_oof_xgb']:.5f} (+{g['dif_oof']:.5f}) y en prueba {g['auc_test']:.5f} frente a {g['auc_test_xgb']:.5f} "
             f"(+{g['dif_test']:.5f}); gana {g['pliegues_ganados']} de 10 pliegues y las cuatro pruebas quedan por debajo de "
             f"0.05 tras Holm.")
else:
    m = R.sort_values('dif_oof', ascending=False).iloc[0]
    texto = (f"**Ningún candidato supera a XGBoost con significancia.** El mejor fue `{m['candidato']}`: AUC fuera de pliegue "
             f"{m['auc_oof']:.5f} frente a {m['auc_oof_xgb']:.5f} ({m['dif_oof']:+.5f}) y prueba {m['auc_test']:.5f} frente a "
             f"{m['auc_test_xgb']:.5f} ({m['dif_test']:+.5f}). Se reporta como resultado negativo, sin cambiar las reglas.")
display(Markdown(texto))"""),
    md("### Lectura de los resultados (generada a partir de los números)"),
    code("""mejor_ind = comp.iloc[0]
lineas = [f"- Compuerta: {len(comp)} algoritmos evaluados; pasan {int(comp.pasa.sum())}. El de mayor AUC de diseño fue "
          f"#{mejor_ind.n} {mejor_ind.algoritmo} ({mejor_ind.auc_diseno:.5f}, {mejor_ind.dif_diseno:+.5f} frente a XGBoost)."]
gd = comp[comp.dif_disjunta > 0]
if len(gd):
    lineas.append('- Superan a XGBoost en la partición disjunta (pero no en las dos): ' +
                  ', '.join(f"#{r.n} {r.variante} ({r.dif_disjunta:+.4f})" for r in gd.itertuples()) + '.')
for r in R.itertuples():
    lineas.append(f"- `{r.candidato}`: OOF {r.dif_oof:+.5f}, prueba {r.dif_test:+.5f}, pliegues ganados {r.pliegues_ganados}/10 "
                  f"-> {r.veredicto}.")
for n in ('sl', 'sl2', 'sl3', 'bstacking'):
    p = os.path.join(RES, f'superlearner_{n}.json')
    if os.path.exists(p):
        w = {k: v for k, v in leer(f'superlearner_{n}.json')['pesos_test'].items() if abs(v) > 0.05}
        lineas.append(f"- Miembros con más peso en `{n}`: " + ', '.join(f"{k} ({v:.2f})" for k, v in
                      sorted(w.items(), key=lambda kv: -abs(kv[1]))) + '.')
display(Markdown(chr(10).join(lineas)))"""),
    md("""Notas:

- Limitación: el código de ELAR-R (`src/`) no estaba en el repositorio entregado, así que esta comparación usa una
  matriz de 787 variables propia y no las 528 entradas del E3; las cifras no se comparan uno a uno con las de
  los capítulos 3 y 6.

## 11.10 Reproducción

```bash
cd vencer_xgb
python features.py          # matriz de variables (datos en ../datos/cache)
python ajuste_xgb.py        # TPE de XGBoost
python supervisor.py        # ruta recomendada: compuertas, protocolo completo, Super Learner y pruebas (reanudable)
python supervisor2.py       # ampliación: los 21 algoritmos restantes, sl3, bstacking y pruebas finales
```"""),
]

nb = nbf.v4.new_notebook(cells=celdas)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, os.path.join(DIR, "11_vencer_xgboost.ipynb"))
print("ok")
