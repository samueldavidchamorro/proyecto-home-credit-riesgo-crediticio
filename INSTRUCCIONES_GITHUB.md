# Instrucciones: añadir una sección y publicarla en GitHub Pages

Proyecto Home Credit Default Risk. Equipo: Jacobo Londoño, Samuel Chamorro y Jesús David Barrios Valdés.

- Repositorio: https://github.com/samueldavidchamorro/proyecto-home-credit-riesgo-crediticio
- Página (GitHub Pages): https://samueldavidchamorro.github.io/proyecto-home-credit-riesgo-crediticio/

## 1. Qué trae el .zip

| Ruta | Contenido |
|---|---|
| `Proyecto_Home_Credit_completo.ipynb` | **El cuaderno único**: todas las secciones unidas, con sus salidas |
| `docs/index.html` | La misma página que publica GitHub Pages |
| `orden_cuaderno.json` | Lista ordenada de las secciones que se unen |
| `construir_cuaderno.py` | Une las secciones y regenera `docs/index.html` |
| `Entregable4/`, `Entregable3_anexo/` | Cuadernos fuente de los entregables 4 y 3 |
| `vencer_xgb/` | Parte II: intento de superar a XGBoost (código, resultados, cuaderno `11_vencer_xgboost.ipynb`) |
| `datos/` | Datos crudos de Kaggle (CSV) y `datos/cache/*.parquet` (3.4 GB). **No están en GitHub** (pesan demasiado) |
| `.git/` | Historial del repositorio, ya conectado a GitHub |

## 2. Preparar el entorno (una vez)

```powershell
cd ruta\donde\descomprimio\ENTREGABLE_FINAL_ML
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Instale también GitHub CLI (https://cli.github.com) y Git, y autentíquese:

```powershell
gh auth login
```

Para poder hacer *push* necesita ser colaborador del repositorio. El dueño (samueldavidchamorro) lo invita desde
GitHub → *Settings* → *Collaborators* → *Add people*. Otra opción es hacer un *fork* y abrir un *pull request*.

## 3. Añadir una sección nueva

1. Cree su cuaderno, por ejemplo `Entregable5/01_nueva_seccion.ipynb`. La **primera celda de markdown debe empezar
   con un título** (`# 12. Nombre de la sección`): ese título aparece en el índice del cuaderno único.
2. Ejecútelo hasta el final para que guarde sus salidas, o márquelo para que se ejecute al unir (paso 3).
3. Añádalo a `orden_cuaderno.json`, en la posición donde quiere que aparezca:

   ```json
   {"parte": "Parte III. Entregable 5", "archivo": "Entregable5/01_nueva_seccion.ipynb", "ejecutar": false}
   ```

   - `parte`: encabezado del bloque. Las entradas seguidas con la misma `parte` quedan bajo un mismo título.
   - `ejecutar`: `true` si quiere que `construir_cuaderno.py` lo ejecute antes de unir. El cuaderno se ejecuta con su
     propia carpeta como directorio de trabajo.
4. Regenere el cuaderno único y la página:

   ```powershell
   python construir_cuaderno.py
   ```

   Use `python construir_cuaderno.py --sin-ejecutar` para unir sin volver a ejecutar nada (más rápido).
   Nota: la Parte II (`vencer_xgb/11_vencer_xgboost.ipynb`) está marcada con `ejecutar: true` y necesita
   `vencer_xgb/cache/` (incluida en el .zip).
5. Revise el resultado abriendo `docs/index.html` en el navegador.

## 4. Subir los cambios a GitHub

```powershell
git status
git add -A
git commit -m "Añade la sección X"
git push origin main
```

`.gitignore` ya excluye `datos/` y `vencer_xgb/cache/X.npy`, así que los datos crudos no se suben. GitHub rechaza
archivos de más de 100 MB: si su sección genera archivos grandes, añádalos a `.gitignore`.

## 5. GitHub Pages

La página ya está configurada para publicarse desde la rama `main`, carpeta `/docs`. Cada `git push` que cambie
`docs/index.html` la actualiza sola en 1–2 minutos (el progreso se ve en la pestaña *Actions* del repositorio).

Si algún día hay que volver a configurarla: GitHub → *Settings* → *Pages* → *Build and deployment* →
*Source: Deploy from a branch* → *Branch: main*, carpeta */docs* → *Save*. Desde la terminal:

```powershell
gh api -X POST repos/samueldavidchamorro/proyecto-home-credit-riesgo-crediticio/pages -f "source[branch]=main" -f "source[path]=/docs"
```

## 6. Reproducir la Parte II (opcional, ~5 h con GPU)

```powershell
cd vencer_xgb
python features.py        # construye cache/X.npy a partir de ../datos/cache
python ajuste_xgb.py      # TPE de XGBoost
python supervisor.py      # compuertas, protocolo completo, Super Learner y pruebas (reanudable)
```

Si los datos están en otra carpeta, fije `HOME_CREDIT_DATOS` con la ruta de la carpeta que contiene los `.parquet`.
