"""Compuerta de cribado para los algoritmos de la ampliación (modelos.py): TPE de 20 evaluaciones en la partición
de diseño y evaluación de la mejor configuración en la disjunta. Mismas reglas que ajuste_propio.py.

Uso: python ajuste_general.py <nombre>      (nombres en modelos.CATALOGO)
"""
import sys
import time
import numpy as np
import optuna
from comun import cargar, particiones, guardar_json, leer_json
import modelos

optuna.logging.set_verbosity(optuna.logging.WARNING)


def evaluar(nombre, p, Xtr, ytr, Xva, yva):
    m = modelos.construir(nombre, p)
    m.fit(Xtr, ytr, Xva, yva)
    return float(m.mejor_auc), m.mejor_it


if __name__ == "__main__":
    nombre = sys.argv[1]
    n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 20  # solo para pruebas de humo; el protocolo usa 20
    _, espacio, tipo = modelos.CATALOGO[nombre]
    X, y = cargar()
    P = particiones()
    t0 = time.time()
    Xtr, Xva = np.asarray(X[P["d_tr"]]), np.asarray(X[P["d_va"]])
    ytr, yva = y[P["d_tr"]], y[P["d_va"]]
    estudio = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))

    def obj(t):
        p = espacio(t)
        a, it = evaluar(nombre, p, Xtr, ytr, Xva, yva)
        t.set_user_attr("it", it)
        t.set_user_attr("params_modelo", p)
        print(f"[{nombre}] trial {t.number} auc {a:.5f} it {it} t={time.time() - t0:.0f}s", flush=True)
        return a

    estudio.optimize(obj, n_trials=n_trials)
    mejor = estudio.best_trial
    p = mejor.user_attrs["params_modelo"]
    Xj, Xjv = np.asarray(X[P["j_tr"]]), np.asarray(X[P["j_va"]])
    a_j, it_j = evaluar(nombre, p, Xj, y[P["j_tr"]], Xjv, y[P["j_va"]])
    xgb = leer_json("xgb_ajuste.json")
    out = dict(variante=nombre, familia="general", tipo_it=tipo, params=p, it=mejor.user_attrs["it"],
               auc_diseno=mejor.value, auc_disjunta=a_j, it_disjunta=it_j, segundos=time.time() - t0,
               xgb_diseno=xgb["auc_diseno"], xgb_disjunta=xgb["auc_disjunta"])
    out["pasa_compuerta"] = bool(out["auc_diseno"] > out["xgb_diseno"] and out["auc_disjunta"] > out["xgb_disjunta"])
    print(out, flush=True)
    if n_trials == 20:
        guardar_json(f"compuerta_{nombre}.json", out)
