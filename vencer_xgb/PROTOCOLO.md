# Protocolo prerregistrado — intento de superar a XGBoost

Fecha de fijación: 2026-11-21, antes de correr cualquier candidato. Ninguna regla de este
archivo se cambia después de ver resultados; si algo se cambia, se deja anotado abajo
en «Desviaciones», con la razón y la hora.

## Datos e igualdad de condiciones

- `features.py` construye una sola matriz (application + agregados de buró, previas, POS,
  cuotas y tarjeta): 307,507 filas × 787 variables. XGBoost y todos los candidatos
  reciben **exactamente** esa matriz, sin variables adicionales.
- Nota: el código original de ELAR-R (`src/`, `experiments/`) y `resultados_e4/` no
  están en el repositorio. Esta matriz no es la de 528 entradas del E3, así que los
  AUC no son comparables uno a uno con los del proyecto; la comparación válida es la
  interna de este directorio.
- Partición externa: 80 % entrenamiento (246,005) / 20 % prueba (61,502), estratificada,
  semilla 2026.

## Núcleo de los candidatos

- Código propio en `gbdt.py` / `modelos.py`, escrito con la API de NumPy. En GPU se
  ejecuta con CuPy (sustituto de NumPy, mismo código, `xp = cupy`); en CPU, con NumPy.
  No se usa xgboost, lightgbm, catboost, torch, tensorflow ni sklearn.ensemble en el
  núcleo. sklearn solo se usa para particiones (`StratifiedKFold`) y métricas.

## Ajuste (mismo presupuesto)

- XGBoost (GPU, `hist`) y cada familia candidata: TPE de Optuna, **20 evaluaciones**,
  semilla 0, en la partición de diseño (64k entrenamiento / 16k validación, tomadas del
  80 % de entrenamiento, semilla 11). Métrica: AUC en validación. El número de rondas
  se fija con parada temprana (paciencia 100) dentro de esa evaluación.
- En el protocolo completo, el número de rondas = mejor iteración del ajuste
  **× 1.2** (hay más filas), con la misma regla para ambos lados.

## Compuerta de cribado

- Un candidato pasa si su AUC supera al de XGBoost ajustado en **dos** particiones:
  la de diseño y una disjunta (otras 64k/16k, semilla 12, sin filas compartidas con la
  de diseño).

## Protocolo completo (solo para quien pasa la compuerta)

- Validación cruzada estratificada 5 pliegues × 2 repeticiones (semillas 0 y 1) sobre
  el 80 %. Predicciones de prueba = promedio de los 5 modelos de la repetición 0.
- Pruebas (bilaterales, α = 0.05), todas con diferencia candidato − XGBoost:
  1. DeLong sobre las predicciones fuera de pliegue (repetición 0, 246k filas).
  2. DeLong sobre el 20 % de prueba.
  3. t corregida de Nadeau–Bengio sobre las 10 diferencias por pliegue
     (n_prueba/n_entrenamiento = 1/4).
  4. Wilcoxon de rangos con signo sobre las 10 diferencias.
- **Veredicto «SUPERA»**: diferencia > 0 en OOF y en prueba, y las cuatro p < 0.05
  después de la corrección de Holm sobre el número de candidatos que llegaron al
  protocolo completo (Holm se aplica por prueba).
- Para el Super Learner (#1): los pesos en el símplex se estiman por validación cruzada
  sobre las predicciones fuera de pliegue (los pesos del pliegue k se estiman con los
  otros pliegues). Los pesos para prueba se estiman con todo el OOF.

## Regla de parada

- Se recorre la lista en el orden de la ruta recomendada (#4 → #2, #3 → #7, #6, #8, #12,
  … → #1). La búsqueda se detiene en el primer candidato con veredicto «SUPERA».
- Si se agota la lista sin «SUPERA», el resultado se reporta como negativo. No se
  cambian semillas, pliegues ni pruebas para buscar significancia.

## Biblioteca del Super Learner (#1), fijada antes de ver resultados de candidatos

- Miembros: `newton` + las variantes propias con mayor AUC en la partición de diseño
  (hasta 4 miembros en total, entre las que terminen el cribado). Los miembros pasan por
  el protocolo completo aunque no hayan pasado la compuerta individual, porque el
  candidato que se evalúa es el ensamble, no cada miembro.
- Selección de Caruana con reemplazo, 50 rondas, sobre márgenes estandarizados.
- El Super Learner entra en Holm junto con cualquier otro candidato que llegue al
  protocolo completo.

## Segunda ronda (fijada el 2026-11-21 22:50, antes de ver el resultado del Super Learner)

Resultado visto hasta ese momento: ninguna variante individual pasó la compuerta
(todas entre −0.0002 y −0.003 frente a XGBoost). Si `sl` no obtiene «SUPERA»:

- **`sl2` = #1 + #11 (MultiBoosting por semillas)**: biblioteca = todas las variantes
  propias con protocolo completo terminado (las 7 del cribado) más `newton` y
  `hojas_lineales` repetidos con semillas 1 y 2 (mismos hiperparámetros). Misma
  selección de Caruana y mismo cruce de pesos por pliegue.
- `sl` y `sl2` entran juntos en Holm (2 candidatos de ensamble + los que hubieran pasado
  la compuerta).
- Si `sl2` tampoco supera, el resultado se reporta como negativo y se entrega así.

## Desviaciones

(ninguna por ahora)
