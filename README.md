# Proyecto Home Credit Default Risk: riesgo crediticio y modelo original ELAR-R

**Equipo:** Jacobo Londoño, Samuel Chamorro y Jesús David Barrios Valdés.

- **Cuaderno único (todo el proyecto, con salidas):** [`Proyecto_Home_Credit_completo.ipynb`](Proyecto_Home_Credit_completo.ipynb)
- **Página web (GitHub Pages):** https://samueldavidchamorro.github.io/proyecto-home-credit-riesgo-crediticio/
- **Cómo añadir una sección y volver a publicar:** [`INSTRUCCIONES_GITHUB.md`](INSTRUCCIONES_GITHUB.md)

El cuaderno único se arma con `python construir_cuaderno.py` a partir del orden de `orden_cuaderno.json`:
Parte I (Entregable 4), Parte II (intento de superar a XGBoost con los algoritmos del Top 30, carpeta `vencer_xgb/`)
y Anexo (Entregable 3).

## Parte II: intento de superar a XGBoost (`vencer_xgb/`)

Siete variantes de un boosting de Newton propio (NumPy/CuPy) y dos Super Learners, con protocolo prerregistrado
(`vencer_xgb/PROTOCOLO.md`). Resultado: **ninguno supera a XGBoost con significancia**. El mejor, el Super Learner,
queda en 0.78945 de AUC fuera de pliegue frente a 0.78973 de XGBoost, y en 0.79196 frente a 0.79296 en prueba.

---

## Cuadernos fuente

Archivos fuente de la implementación (`.ipynb`) **ya ejecutados**, con todas las salidas y figuras visibles.
Son copias idénticas de los cuadernos de la raíz del repositorio, en el orden del libro (`_toc.yml`).

## Entregable 4 (`Entregable4/`)

| Cuaderno | Contenido |
|---|---|
| `01_correcciones.ipynb` | Trazabilidad de las correcciones pedidas en el Entregable 3 |
| `02_datos.ipynb` | ETL, EDA, declaración de datos y auditoría de fuga |
| `03_benchmarks.ipynb` | Modelos benchmark |
| `06_validacion.ipynb` | Protocolo de validación y pruebas de hipótesis |
| `07_evaluacion_ampliada.ipynb` | Conjuntos adicionales y robustez |
| **`08_dashboard.ipynb`** | **Implementación del dashboard ejecutada**: carga la aplicación Dash, llama a los callbacks de sus tres páginas, muestra las 26 figuras, hace predicciones en vivo con los 6 modelos, explica cada predicción (SHAP), muestra los casos de ELAR-R y valida el servidor y las cifras (62 de 62 chequeos) |
| `10_reproducibilidad.ipynb` | Entorno, semillas, tabla maestra y bibliografía |

Los capítulos 04 (literatura), 05 (modelo original ELAR-R) y 09 (conclusiones) son texto (`.md`) y están en la raíz. El
capítulo 08 tiene además su texto de diseño en `08_dashboard.md`.

## Anexo: Entregable 3 (`Entregable3_anexo/`)

| Cuaderno | Contenido |
|---|---|
| `01_datos_y_linea_base.ipynb` | Datos, preprocesamiento y línea base (genera `datos/`, que usa el dashboard) |
| `02_clasificacion.ipynb` | Estudio de clasificación |
| `03_regresion.ipynb` | Estudio de regresión |
| `04_optimizacion_hiperparametros.ipynb` | Optimización de hiperparámetros |
| `05_optimizacion_computacional.ipynb` | Optimización computacional |
| `06_evaluacion_clasificacion.ipynb` | Evaluación de clasificación |
| `07_evaluacion_regresion.ipynb` | Evaluación de regresión |
| `08_interpretabilidad.ipynb` | Interpretabilidad (SHAP) |
| `09_comparacion_estadistica.ipynb` | Comparación estadística |
| `A1_ejecucion_del_estudio.ipynb` | Ejecución del estudio completo |

## Volver a ejecutarlos

Los cuadernos usan rutas relativas a la raíz del repositorio (`src/`, `datos/`, `resultados/`, ...). Para ejecutarlos de nuevo,
ábralos desde la raíz, no desde esta carpeta. `08_dashboard.ipynb` es la excepción: busca la raíz por su cuenta y corre desde
aquí, con las dependencias de `requirements_dashboard.txt` (Python 3.12) más `ipykernel` y `kaleido==0.1.0.post1` (exporta
las figuras a PNG).

## Relación con el dashboard

- Dashboard: https://dashriesgocrediticio.onrender.com
- Código del dashboard: `app.py`, `pages/` y `src/servicio.py` en la raíz.
- Arquitectura y fuentes de cada cifra: `08_dashboard.md`, sección 8.2.
