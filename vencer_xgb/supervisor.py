"""Supervisor: corre todo el plan de PROTOCOLO.md de forma reanudable y con reintentos.

Cada paso es idempotente (se salta si su salida ya existe; protocolo.py reanuda por pliegue).
Mantiene el equipo despierto mientras corre (SetThreadExecutionState; no cambia ajustes).
Al final llama a entrega.py (cuaderno integrado, GitHub Pages y .zip) si existe.
"""
import ctypes
import datetime
import json
import os
import subprocess
import sys
import time
import numpy as np
from comun import RES, DIR

PY = sys.executable
VARIANTES = ["newton", "hojas_lineales", "heterogeneo", "mvs", "olvidadizo", "dart", "goss"]
ESTADO = os.path.join(RES, "ESTADO.txt")


def despierto():
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception:
        pass


def log(msg):
    linea = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(linea, flush=True)
    with open(ESTADO, "a", encoding="utf-8") as f:
        f.write(linea + "\n")


def correr(args, archivo_log, intentos=3):
    for i in range(intentos):
        despierto()
        with open(os.path.join(RES, archivo_log), "a", encoding="utf-8") as f:
            r = subprocess.run([PY, "-W", "ignore"] + args, stdout=f, stderr=subprocess.STDOUT, cwd=DIR)
        if r.returncode == 0:
            return True
        log(f"FALLO {' '.join(args)} (intento {i + 1}/{intentos}, código {r.returncode}); reintento")
        time.sleep(30)
    log(f"ABANDONO {' '.join(args)} tras {intentos} intentos")
    return False


def protocolo_hecho(m):
    p = os.path.join(RES, f"pred_{m}.npz")
    return os.path.exists(p) and not np.isnan(np.load(p)["auc_pliegues"]).any()


def asegurar_protocolo(m):
    if not protocolo_hecho(m):
        log(f"protocolo completo: {m}")
        correr(["protocolo.py", m], f"log_protocolo_{m}.txt")
    return protocolo_hecho(m)


def compuertas():
    out = {}
    for v in VARIANTES:
        p = os.path.join(RES, f"compuerta_{v}.json")
        if not os.path.exists(p):
            log(f"compuerta: {v}")
            correr(["ajuste_propio.py", v], f"log_compuerta_{v}.txt")
        if os.path.exists(p):
            out[v] = json.load(open(p, encoding="utf-8"))
            log(f"compuerta {v}: diseño {out[v]['auc_diseno']:.5f} disjunta {out[v]['auc_disjunta']:.5f} pasa={out[v]['pasa_compuerta']}")
    return out


def pruebas(cands):
    nombre = "pruebas_" + "_".join(cands) + ".json"
    correr(["pruebas.py"] + cands, "log_pruebas.txt")
    R = json.load(open(os.path.join(RES, nombre), encoding="utf-8"))
    for r in R:
        log(f"PRUEBAS {r['candidato']}: OOF {r['auc_oof']:.5f} vs {r['auc_oof_xgb']:.5f} | prueba {r['auc_test']:.5f} vs "
            f"{r['auc_test_xgb']:.5f} | p_holm DeLongOOF {r['p_delong_oof_holm']:.4f} DeLongTest {r['p_delong_test_holm']:.4f} "
            f"NB {r['p_nb_holm']:.4f} Wilcoxon {r['p_wilcoxon_holm']:.4f} -> {r['veredicto']}")
    json.dump(R, open(os.path.join(RES, "pruebas_final.json"), "w", encoding="utf-8"), indent=2, default=float)
    return any(r["veredicto"] == "SUPERA" for r in R)


def main():
    despierto()
    log("supervisor iniciado")
    if not protocolo_hecho("xgb"):
        correr(["protocolo.py", "xgb"], "log_protocolo_xgb.txt")
    comp = compuertas()
    pasan = [v for v, c in comp.items() if c["pasa_compuerta"]]
    otros = sorted([v for v in comp if v != "newton"], key=lambda v: -comp[v]["auc_diseno"])
    bib1 = ["newton"] + otros[:3]
    log(f"pasan compuerta: {pasan} | biblioteca sl: {bib1}")
    for m in dict.fromkeys(pasan + bib1):
        asegurar_protocolo(m)
    bib1 = [m for m in bib1 if protocolo_hecho(m)]
    correr(["superlearner.py", "sl"] + bib1, "log_superlearner.txt")
    pasan = [m for m in pasan if protocolo_hecho(m)]
    gano = pruebas(pasan + ["sl"])
    if not gano:
        log("sl no supera: segunda ronda (sl2 = #1 + #11)")
        bib2 = list(comp) + ["newton_s1", "newton_s2", "hojas_lineales_s1", "hojas_lineales_s2"]
        for m in bib2:
            asegurar_protocolo(m)
        bib2 = [m for m in bib2 if protocolo_hecho(m)]
        correr(["superlearner.py", "sl2"] + bib2, "log_superlearner.txt")
        gano = pruebas(pasan + ["sl", "sl2"])
    log(f"BUSQUEDA TERMINADA: {'HAY GANADOR' if gano else 'NINGUNO SUPERA (resultado negativo)'}")
    if os.path.exists(os.path.join(DIR, "entrega.py")):
        log("entrega: cuaderno integrado, GitHub Pages y zip")
        correr(["entrega.py"], "log_entrega.txt", intentos=2)
    log("supervisor terminado")


if __name__ == "__main__":
    main()
