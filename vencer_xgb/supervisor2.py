"""Supervisor de la ampliación (PROTOCOLO.md, sección «Ampliación»): los 21 algoritmos restantes del Top 30.

Reanudable e idempotente, con reintentos. Orden:
  1. compuerta de cada algoritmo de modelos.ORDEN; si uno pasa, protocolo completo y pruebas inmediatas
     (Holm sobre la familia: sl, sl2 y los individuales que pasaron); si «SUPERA», se detiene.
  2. biblioteca sl3 = 11 predicciones propias previas + nuevos con AUC de diseño ≥ 0.7695; protocolo completo de los
     nuevos miembros; sl3 (Caruana) y bstacking (#15, logística) sobre la misma biblioteca.
  3. pruebas finales sobre la familia completa y entrega (cuaderno, GitHub Pages, zip).
"""
import json
import os
from supervisor import log, correr, protocolo_hecho, asegurar_protocolo, pruebas, despierto, DIR, RES
import modelos

PREVIOS = ["newton", "hojas_lineales", "heterogeneo", "mvs", "olvidadizo", "dart", "goss",
           "newton_s1", "newton_s2", "hojas_lineales_s1", "hojas_lineales_s2"]
UMBRAL_BIB = 0.7695


def main():
    despierto()
    log("supervisor2 (ampliación, 21 algoritmos) iniciado")
    comp, pasan = {}, []
    gano = False
    for v in modelos.ORDEN:
        p = os.path.join(RES, f"compuerta_{v}.json")
        if not os.path.exists(p):
            log(f"compuerta: {v}")
            correr(["ajuste_general.py", v], f"log_compuerta_{v}.txt", intentos=2)
        if not os.path.exists(p):
            log(f"compuerta {v}: SIN RESULTADO (falló dos veces)")
            continue
        comp[v] = json.load(open(p, encoding="utf-8"))
        c = comp[v]
        log(f"compuerta {v}: diseño {c['auc_diseno']:.5f} disjunta {c['auc_disjunta']:.5f} pasa={c['pasa_compuerta']}")
        if c["pasa_compuerta"]:
            if asegurar_protocolo(v):
                pasan.append(v)
                if pruebas(["sl", "sl2"] + pasan):
                    gano = True
                    log(f"GANADOR: {v}; la búsqueda se detiene (regla de parada)")
                    break
    if not gano:
        nuevos = [v for v, c in comp.items() if c["auc_diseno"] >= UMBRAL_BIB]
        log(f"biblioteca sl3: previos + {nuevos}")
        for v in nuevos:
            asegurar_protocolo(v)
        bib = [m for m in PREVIOS + nuevos if protocolo_hecho(m)]
        correr(["superlearner.py", "sl3"] + bib, "log_superlearner.txt")
        correr(["superlearner.py", "bstacking", "--logistica"] + bib, "log_superlearner.txt")
        gano = pruebas(["sl", "sl2"] + pasan + ["sl3", "bstacking"])
    log(f"AMPLIACIÓN TERMINADA: {'HAY GANADOR' if gano else 'NINGUNO SUPERA (resultado negativo)'}")
    if os.path.exists(os.path.join(DIR, "entrega.py")):
        log("entrega: cuaderno integrado, GitHub Pages y zip")
        correr(["entrega.py"], "log_entrega.txt", intentos=2)
    log("supervisor2 terminado")


if __name__ == "__main__":
    main()
