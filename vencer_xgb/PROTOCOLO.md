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


## Ampliación: los 21 algoritmos restantes del Top 30 (fijada el 2026-11-22 06:00, antes de correrlos)

Motivo: el pedido original era probar **todos** los algoritmos del documento; la primera búsqueda solo cubrió
#1, #2, #3, #4, #6, #7, #11, #12 y #26. Resultados ya vistos al fijar esta sección: los de la primera búsqueda
(ninguno supera; mejor `sl`, −0.0003 OOF). Las reglas de abajo no se cambian después de ver los nuevos números.

Candidatos nuevos (código propio en `gbdt.py` y `modelos.py`, NumPy/CuPy, sin librerías de ensambles):

| # | Nombre en el código | Implementación |
|---|---|---|
| 5 | `ktboost` | Boosting de Newton que en cada ronda elige entre un árbol y una ridge sobre rasgos aleatorios de Fourier (núcleo RBF), el de mayor reducción de la pérdida |
| 8 | `rotaciones` | Boosting de Newton con vistas: cada árbol ve las variables originales más bloques de variables continuas rotadas al azar (rotación ortogonal por bloques, tras estandarizar) |
| 9 | `oblicuos` | Igual, pero las vistas añaden proyecciones dispersas ±1 de 2–4 variables estandarizadas (cortes oblicuos tipo SPORF) |
| 10 | `rgf` | Boosting de Newton con paso totalmente correctivo: cada cierto número de rondas reajusta las hojas de todos los árboles (descenso por coordenadas de Newton) |
| 13 | `infiniteboost` | F = c · promedio de árboles; c (capacidad) se ajusta con un paso de Newton por ronda |
| 14 | `lce` | Ensamble en cascada local: por bolsa, un boosting raíz, un corte, y en cada hijo un boosting que parte del margen del padre y lo usa como variable |
| 15 | `bstacking` | Apilamiento con metamodelo logístico L2 (IRLS propio) sobre los márgenes OOF de la biblioteca, con pesos cruzados por pliegue como el Super Learner |
| 16 | `rankboost` | Boosting de Newton con pérdida logística por pares (positivo, negativo muestreados), que optimiza el AUC |
| 17 | `trboost` | Boosting con región de confianza: hoja = −G/(H+λ+μ), μ se adapta con el cociente reducción real / reducción predicha |
| 18 | `nesterov` | Boosting acelerado de Nesterov (Biau, Cadre y Rouvière 2019), dos sucesiones y pesos por árbol |
| 19 | `focal` | Boosting de Newton con pérdida focal (γ ajustable); gradiente y hessiano por diferencias centrales |
| 20 | `bosque_profundo` | Cascada de bosques (aleatorio + extremadamente aleatorio) con vectores de clase OOF de 3 pliegues; niveles hasta que deja de mejorar |
| 21 | `bosque_rotacion` | Bosque de rotación: cada árbol sobre PCA de subconjuntos aleatorios de variables continuas en una muestra bootstrap |
| 22 | `extra_trees` | Árboles extremadamente aleatorios (corte al azar por nodo y variable, mtry por nodo) |
| 23 | `spe` | Ensamble de ritmo propio: submuestreo de la clase mayoritaria por dureza con factor de ritmo, base = boosting propio pequeño |
| 24 | `des` | Selección dinámica: grupo de boostings propios por bootstrap; para cada fila, pesos según la competencia local (k vecinos en un conjunto DSEL de 20 %, espacio PCA) |
| 25 | `lmt` | Árbol de modelo logístico: árbol poco profundo y, en cada hoja, regresión logística L2 (IRLS) encogida hacia la logística de la raíz |
| 27 | `bart` | BART probit: suma de árboles con MCMC (crecer/podar, aumento de datos de Albert–Chib), media posterior |
| 28 | `psplines` | Boosting por componentes con P-splines (base B-spline cúbica sobre las cubetas de cada variable, penalización de diferencias) |
| 29 | `ga2m` | EBM: árboles de una variable en ciclo y luego pares de interacción elegidos por ganancia |
| 30 | `ngboost` | Boosting por gradiente natural para Bernoulli (árboles de regresión al gradiente natural + búsqueda lineal) |

Reglas:

1. **Compuerta igual que antes**: TPE de Optuna, 20 evaluaciones, semilla 0, partición de diseño; la mejor
   configuración se evalúa en la disjunta; pasa si supera a XGBoost en las dos. Los boostings usan el mismo espacio
   base que XGBoost más sus parámetros propios; los bosques, BART, LMT, DES y SPE usan su propio espacio (declarado
   en `modelos.py`).
2. **Biblioteca `sl3`**: las 11 predicciones propias ya calculadas + todo candidato nuevo con AUC de diseño
   ≥ 0.7695 (XGBoost − 0.01), que pasa por el protocolo completo aunque no pase la compuerta. Caruana, 50 rondas,
   mismos pesos cruzados. `bstacking` (#15) usa exactamente la misma biblioteca.
3. **Protocolo completo** para quien pase la compuerta, para los miembros de `sl3`, y para `sl3` y `bstacking`.
4. **Holm** sobre la familia completa del estudio: `sl`, `sl2`, `sl3`, `bstacking` y todo candidato individual que
   pase la compuerta. Veredicto «SUPERA» igual que antes (las cuatro p ajustadas < 0.05 y diferencias > 0).
5. **Parada**: si un candidato que pasa la compuerta obtiene «SUPERA», la búsqueda se detiene ahí. Si no, se
   recorren los 21 y se cierra con `sl3` y `bstacking`. Si nada supera, se reporta como negativo.

## Desviaciones

- 2026-11-22: la primera búsqueda se cerró sin probar 21 de los 30 algoritmos, aunque el pedido era probarlos todos.
  Se corrige con la ampliación de arriba, fijada antes de correrlos.
- 2026-11-22 17:30: la compuerta del bosque profundo (#20) se interrumpió en su 5.ª evaluación porque la GPU (4 GB)
  se desbordaba a memoria compartida del sistema (7,6 GB por PCIe) y una evaluación llegó a tardar 4,7 h. Se añadió
  un tope al pool de memoria de CuPy y la búsqueda de cortes por trozos de nodos (`gbdt.py`); ninguno de los dos
  cambia el modelo, el espacio de búsqueda ni la semilla. La compuerta se repite completa desde cero.
- 2026-11-22/23: DES (#24) se quedó sin memoria con ese tope (compuerta y protocolo completo). Se redujo el bloque de
  consultas de la búsqueda de vecinos (4096 → 256 filas) y se libera la caché del pool antes; el cálculo es el mismo.
  Los intentos fallidos no produjeron resultados; cuentan solo los intentos completos.
- Nota (no es desviación): en RGF (#10) la compuerta paró en la ronda 190, justo antes de su primer paso totalmente
  correctivo (cada 191); en el protocolo completo (228 rondas) ese paso sí ocurre y la AUC baja (OOF 0.739). Se
  reporta tal cual, según la regla de rondas × 1.2 fijada de antemano.
