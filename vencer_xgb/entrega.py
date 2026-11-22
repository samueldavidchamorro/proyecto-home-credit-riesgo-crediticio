"""Entrega final: cuaderno único -> repositorio en GitHub -> GitHub Pages -> .zip con todo (incluidos los datos).

Uso (desde cualquier carpeta):  python vencer_xgb/entrega.py [--sin-zip] [--sin-github]
"""
import json
import os
import subprocess
import sys
import zipfile

DIR = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(DIR)
USUARIO = "samueldavidchamorro"
REPO = "proyecto-home-credit-riesgo-crediticio"
DATOS = os.environ.get("HOME_CREDIT_DATOS_CRUDOS",
                       r"C:\Users\samit\OneDrive\Escritorio\UNIVERSIDAD\2026\2026-3\MACHINE LEARNING\03_Proyecto_Home_Credit\datos")
ZIP = os.path.join(os.path.dirname(RAIZ), "ENTREGA_HOME_CREDIT.zip")
PAGES = f"https://{USUARIO}.github.io/{REPO}/"


def sh(*cmd, cwd=RAIZ, ok=False):
    print(">", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    print((r.stdout + r.stderr).strip()[-2000:], flush=True)
    if r.returncode and not ok:
        raise SystemExit(f"falló: {' '.join(cmd)}")
    return r


def cuaderno():
    sh(sys.executable, "construir_seccion.py", cwd=DIR)
    sh(sys.executable, "-W", "ignore", "construir_cuaderno.py")


def github():
    remotos = sh("git", "remote", ok=True).stdout.split()
    if "origin" in remotos and "original" not in remotos:
        url = sh("git", "remote", "get-url", "origin").stdout.strip()
        if "jacob0900" in url:
            sh("git", "remote", "rename", "origin", "original")
    if sh("gh", "repo", "view", f"{USUARIO}/{REPO}", ok=True).returncode:
        sh("gh", "repo", "create", f"{USUARIO}/{REPO}", "--public",
           "--description", "Home Credit Default Risk: proyecto completo en un solo cuaderno (GitHub Pages)")
    if "origin" not in sh("git", "remote", ok=True).stdout.split():
        sh("git", "remote", "add", "origin", f"https://github.com/{USUARIO}/{REPO}.git")
    sh("git", "add", "-A")
    if sh("git", "diff", "--cached", "--quiet", ok=True).returncode:
        sh("git", "commit", "-m", "Proyecto completo: cuaderno único, intento de superar a XGBoost y página de GitHub Pages")
    sh("git", "branch", "-M", "main")
    sh("git", "push", "-u", "origin", "main")
    r = sh("gh", "api", "-X", "POST", f"repos/{USUARIO}/{REPO}/pages",
           "-f", "source[branch]=main", "-f", "source[path]=/docs", ok=True)
    if r.returncode:
        sh("gh", "api", "-X", "PUT", f"repos/{USUARIO}/{REPO}/pages",
           "-f", "source[branch]=main", "-f", "source[path]=/docs", ok=True)


def comprimir():
    excluir = {"__pycache__", "datos_hc", ".ipynb_checkpoints"}
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        base = os.path.basename(RAIZ)
        for origen, prefijo in ((RAIZ, base), (DATOS, os.path.join(base, "datos"))):
            for d, subdirs, archivos in os.walk(origen):
                subdirs[:] = [s for s in subdirs if s not in excluir]
                for a in archivos:
                    p = os.path.join(d, a)
                    z.write(p, os.path.join(prefijo, os.path.relpath(p, origen)))
    print("zip:", ZIP, f"{os.path.getsize(ZIP) / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    cuaderno()
    if "--sin-github" not in sys.argv:
        github()
    json.dump({"repositorio": f"https://github.com/{USUARIO}/{REPO}", "pages": PAGES, "zip": ZIP},
              open(os.path.join(RAIZ, "ENTREGA_INFO.json"), "w"), indent=2)
    if "--sin-zip" not in sys.argv:
        comprimir()
    print("listo:", PAGES)
