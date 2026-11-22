"""Integra todas las secciones del proyecto en UN solo cuaderno y genera la página de GitHub Pages.

Uso (desde la raíz del proyecto):
    python construir_cuaderno.py              # ejecuta las secciones marcadas y une todo
    python construir_cuaderno.py --sin-ejecutar

Lee el orden de `orden_cuaderno.json`. Para añadir una sección nueva, agregue una entrada a esa lista
(ver INSTRUCCIONES_GITHUB.md). Salidas:
    Proyecto_Home_Credit_completo.ipynb   (cuaderno único, con salidas)
    docs/index.html                       (la página que publica GitHub Pages)
"""
import copy
import json
import os
import sys
import uuid
import nbformat
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor

RAIZ = os.path.dirname(os.path.abspath(__file__))


def ejecutar(ruta):
    nb = nbformat.read(ruta, as_version=4)
    ep = ExecutePreprocessor(timeout=3600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": os.path.dirname(ruta)}})
    nbformat.write(nb, ruta)
    print("  ejecutado:", os.path.relpath(ruta, RAIZ))


def main():
    cfg = json.load(open(os.path.join(RAIZ, "orden_cuaderno.json"), encoding="utf-8"))
    sin_ejecutar = "--sin-ejecutar" in sys.argv
    indice, celdas, parte_actual = [], [], None
    for s in cfg["secciones"]:
        ruta = os.path.join(RAIZ, s["archivo"])
        if not os.path.exists(ruta):
            print("  AVISO: no existe", s["archivo"], "(se omite)")
            continue
        if s.get("ejecutar") and not sin_ejecutar:
            ejecutar(ruta)
        nb = nbformat.read(ruta, as_version=4)
        if s["parte"] != parte_actual:
            parte_actual = s["parte"]
            celdas.append(nbformat.v4.new_markdown_cell(f"---\n\n# {parte_actual}"))
            indice.append(f"\n**{parte_actual}**\n")
        titulo = next((c.source.splitlines()[0].lstrip("# ").strip() for c in nb.cells
                       if c.cell_type == "markdown" and c.source.strip().startswith("#")), s["archivo"])
        indice.append(f"- {titulo}  \n  <small>`{s['archivo']}`</small>")
        celdas.append(nbformat.v4.new_markdown_cell(f"<small>Fuente: `{s['archivo']}`</small>"))
        for c in nb.cells:
            c = copy.deepcopy(c)
            c["id"] = uuid.uuid4().hex[:12]
            celdas.append(c)
    portada = nbformat.v4.new_markdown_cell(
        f"# {cfg['titulo']}\n\n**Equipo:** {cfg['equipo']}\n\n"
        "Este cuaderno reúne en un solo archivo todas las secciones del proyecto, con sus salidas. "
        "Cada sección indica el cuaderno de origen.\n\n## Contenido\n" + "\n".join(indice))
    final = nbformat.v4.new_notebook(cells=[portada] + celdas)
    final.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    final.metadata["language_info"] = {"name": "python"}
    salida = os.path.join(RAIZ, cfg["salida"])
    nbformat.write(final, salida)
    print("cuaderno único:", cfg["salida"], f"({len(final.cells)} celdas)")
    html, _ = HTMLExporter(template_name="lab").from_notebook_node(final)
    os.makedirs(os.path.join(RAIZ, "docs"), exist_ok=True)
    with open(os.path.join(RAIZ, "docs", "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
    open(os.path.join(RAIZ, "docs", ".nojekyll"), "w").close()
    print("página:", "docs/index.html", f"({len(html) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
