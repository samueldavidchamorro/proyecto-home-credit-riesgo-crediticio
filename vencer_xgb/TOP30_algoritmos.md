# Top 30 de algoritmos originales para intentar vencer a XGBoost con significancia

Proyecto: Home Credit Default Risk, modelo original ELAR-R. Equipo: Jacobo Londoño, Samuel Chamorro y Jesús David Barrios Valdés.

## 1. Punto de partida (datos reales del proyecto)

- **Comparación en igualdad de condiciones (E3, mismas 528 entradas).** XGBoost tiene 0.7877 de AUC fuera de pliegue y ELAR-R tiene 0.7838. La diferencia es −0.0039 y la t de Nadeau–Bengio da p = 0.0396 (`resultados_e4/e10_igualdad/pruebas_elar.json`). El veredicto fue «NO SUPERA».
- **Lo que falta.** Para ganarle con significancia en las cuatro pruebas (DeLong fuera de pliegue, DeLong en prueba, Nadeau–Bengio y Wilcoxon), ELAR-R tendría que quedar unos +0.004 por encima de XGBoost, en ≈0.792. Le faltan unos +0.008.
- **Tres mejoras ya probadas en la iteración 8**: `mayorante="agudo"`, `reajuste_final=True` y `ponderar_clases=False` (`resultados_e4/compuerta8/`).

  | Partición | ELAR-R base | ELAR-R it8 | XGBoost |
  |---|---|---|---|
  | Diseño 64k/16k | 0.76582 | 0.76597 (+0.00015) | 0.76534 |
  | Disjunta | — | 0.75128 | 0.75467 (diferencia −0.0034; compuerta ROJO) |

  La ganancia estimada de +0.004 a +0.009 no apareció. La causa de la brecha no es la curvatura ni los pesos de clase.

## 2. Qué permiten los lineamientos del profesor Lihki Rubio

- **R2.** Clase propia con `BaseEstimator` y `ClassifierMixin`, con solo NumPy y SciPy en el núcleo. Están prohibidos como núcleo `xgboost`, `lightgbm`, `catboost`, `torch`, `tensorflow` y `sklearn.ensemble`.
- **R1.** Novedad en la aplicación.
- **R3.** Superioridad y validez estadística.

Por eso quedan descartados como modelo original el apilamiento de XGBoost, CatBoost y LightGBM; AutoGluon; y usar XGBoost o CatBoost como vista h5. Todos usan librerías de terceros como núcleo; solo sirven como referencia del techo alcanzable.

Todo lo de abajo lo tiene que programar el equipo en NumPy y SciPy. La fuente da la idea publicada; lo original es la combinación y su aplicación.

## 3. Cómo leer la tabla

- **Probabilidad:** juicio, no medición. Ninguna fuente verificada muestra que estos métodos le ganen con significancia a un XGBoost ajustado en Home Credit.
- **Fuente:** solo las verificadas. Cuando la fuente no respalda la afirmación contra XGBoost, se dice en las notas.
- **Coste:** coste aproximado de programarlo y correrlo.

## 4. Top 30

| # | Algoritmo original (qué programar) | Fuente verificada | Qué dice la fuente | Probabilidad | Coste | Notas |
|---|---|---|---|---|---|---|
| 1 | Super Learner con biblioteca de boostings propios heterogéneos, con selección de Caruana o pesos en el símplex | Caruana et al., ICML 2004; van der Laan, Polley y Hubbard, SAGMB 2007; Erickson et al., TabArena, NeurIPS 2025; Shwartz-Ziv y Armon, Inf. Fusion 2022 | Los ensambles entre modelos hacen avanzar el estado del arte tabular | Media (la mejor) | Alto | Cambiar la biblioteca por boostings propios fuertes y distintos |
| 2 | GBDT con hojas lineales | Shi, Li y Li, IJCAI 2019 | Mejor que XGBoost, LightGBM y CatBoost en datos numéricos densos | Media | Medio | La hoja pasa a ser una ridge local |
| 3 | Boosting de Newton heterogéneo (profundidad aleatoria / rasgos de Fourier) | Parnell et al., SnapBoost, NeurIPS 2020 | Mejor generalización en OpenML y Kaggle | Media | Medio | Convergencia lineal demostrada |
| 4 | Boosting de Newton propio con el mismo presupuesto TPE que XGBoost | Sigrist, ESWA 2021; Chen y Guestrin, KDD 2016 | Newton supera al gradiente en la mayoría de conjuntos | Media-baja (requisito) | Bajo | Si no llega a XGBoost, nada de lo demás alcanzará |
| 5 | Boosting árbol + núcleo (Nyström / Fourier) | Sigrist, KTBoost, 2021 | Supera a solo árboles y solo núcleos | Media-baja | Alto | Kernel aproximado |
| 6 | Boosting ordenado con árboles olvidadizos | Prokhorenkova et al., CatBoost, NeurIPS 2018 | Corrige el desplazamiento de predicción | Media-baja | Medio | Diversidad para #1 |
| 7 | DART | Rashmi y Gilad-Bachrach, AISTATS 2015 | Supera a MART | Media-baja | Bajo | Comparación contra MART |
| 8 | Boosting con rotaciones aleatorias | Blaser y Fryzlewicz, JMLR 2016 | Ganancias en ensambles de árboles | Media-baja | Bajo | Estandarizar antes de rotar |
| 9 | Cortes oblicuos dispersos | Tomita et al., SPORF, JMLR 2020 | Mejora en más de 100 problemas (bosques) | Media-baja | Medio | Solo sugiere la extensión al boosting |
| 10 | RGF totalmente correctivo | Johnson y Zhang, TPAMI 2014 | Más precisión que gradient boosting | Media-baja | Alto | Anterior a XGBoost |
| 11 | Boosting de Newton embolsado (MultiBoosting) | Webb, Machine Learning 2000 | Menor error que AdaBoost o wagging | Baja | Bajo | Poca varianza que quitar |
| 12 | Muestreo de varianza mínima (MVS) | Ibragimov y Gusev, NeurIPS 2019 | Mejora frente a otros muestreos | Baja | Bajo | Muestreo por defecto de CatBoost |
| 13 | InfiniteBoost | Rogozhnikov y Likhomanenko, arXiv 2017 | Converge al crecer el número de árboles | Baja | Medio | Preprint |
| 14 | Ensamble en cascada local (LCE) | Fauvel et al., arXiv 2005.03645 | Supera al estado del arte en UCI | Baja | Medio | Sin confirmar contra XGBoost |
| 15 | bstacking propio | Xia et al., ESWA 2018 | Supera referencias en riesgo crediticio | Baja-media | Medio | Solo aprendices propios |
| 16 | Boosting por pares que maximiza AUC (RankBoost) | Freund et al., JMLR 2003; Rudin y Schapire, JMLR 2009 | AUC equivalentes a AdaBoost | Baja | Bajo | Poca ganancia esperada |
| 17 | TRBoost | Luo et al., Appl. Intell. 2023 | Competitivo con segundo orden | Baja | Bajo | Empate |
| 18 | Boosting acelerado de Nesterov | Biau, Cadre y Rouvière, ML 2019 | Modelos más dispersos | Baja | Bajo | No mejora AUC |
| 19 | Pérdida focal | Wang 2019; Luo et al. 2023, 2024 | Mejoras ante desbalance | Baja | Bajo | Apunta al umbral, no al AUC |
| 20 | Bosque profundo en cascada | Zhou y Feng, IJCAI 2017 | Competitivo con redes profundas | Baja | Alto | Muy pesado |
| 21 | Bosque de rotación | Rodríguez, Kuncheva y Alonso, TPAMI 2006 | Mejor que bagging y RF en UCI | Baja | Medio | Anterior a XGBoost |
| 22 | Árboles extremadamente aleatorios propios | Geurts, Ernst y Wehenkel, ML 2006 | Aleatoriza variable y corte | Baja | Bajo | Diversidad para #1 |
| 23 | Ensamble de ritmo propio (SPE) | Liu et al., ICDE 2020 | Robusto ante desbalance extremo | Baja | Medio | Descarta datos |
| 24 | Selección dinámica de ensambles | Feng et al., ASC 2018 | Superioridad en riesgo crediticio | Baja | Medio | Requiere vecindarios |
| 25 | Árboles de modelo logístico (LMT) | Landwehr, Hall y Frank, ML 2005 | Árboles + regresión logística | Muy baja | Medio | — |
| 26 | GOSS | Ke et al., LightGBM, NeurIPS 2017 | Más rápido, casi igual precisión | Muy baja en AUC | Bajo | Paga más rondas |
| 27 | BART probit | Chipman, George y McCulloch, AOAS 2010 | Suma bayesiana de árboles | Muy baja | Muy alto | MCMC inviable |
| 28 | Boosting por componentes con P-splines | Bühlmann y Hothorn, Stat. Sci. 2007 | Boosting como regularización | Muy baja | Bajo | Interpretabilidad |
| 29 | GA²M/EBM mejorado | Nori et al., InterpretML 2019 | Caja blanca casi tan precisa | Muy baja | Bajo | Interpretabilidad |
| 30 | NGBoost | Duan et al., ICML 2020 | Igual o mejor en predicción probabilística | Muy baja | Medio | Pensado para regresión |

## 5. Ruta recomendada

1. Compuerta 0 (#4): booster de Newton propio, ajustado con el mismo TPE de 20 evaluaciones que XGBoost, en la partición de diseño 64k/16k.
2. Innovaciones de hoja y de aprendiz (#2, #3), de una en una, en la misma compuerta.
3. Biblioteca (#1): las mejores variantes, más #6, #7 y #8, combinadas en el símplex.
4. Protocolo completo solo si pasa la compuerta en la partición disjunta.

## 6. Reglas estadísticas fijadas antes de correr

- Número de repeticiones de la validación cruzada, ROPE, pruebas y corrección de Holm, escritos antes de ver resultados.
- Cambiar cualquiera de esas reglas tras ver los números cuenta como p-hacking.
- Más repeticiones dan potencia, pero solo ayudan si la diferencia real es positiva.
- La ingeniería de variables no da ventaja en igualdad de condiciones.

## 7. Fuentes principales

Parnell et al. 2020 (SnapBoost); Sigrist 2021 (KTBoost; Gradient and Newton boosting); Chen y Guestrin 2016 (XGBoost); Shi, Li y Li 2019 (PL Trees); Prokhorenkova et al. 2018 (CatBoost); Rashmi y Gilad-Bachrach 2015 (DART); Blaser y Fryzlewicz 2016; Tomita et al. 2020 (SPORF); Johnson y Zhang 2014 (RGF); Webb 2000 (MultiBoosting); Ibragimov y Gusev 2019 (MVS); Ke et al. 2017 (LightGBM/GOSS); Caruana et al. 2004 (Ensemble selection); van der Laan, Polley y Hubbard 2007 (Super Learner); Erickson et al. 2025 (TabArena); Shwartz-Ziv y Armon 2022; Grinsztajn, Oyallon y Varoquaux 2022; Nadeau y Bengio 2003; Bouckaert y Frank 2004; DeLong, DeLong y Clarke-Pearson 1988; Sun y Xu 2014.
