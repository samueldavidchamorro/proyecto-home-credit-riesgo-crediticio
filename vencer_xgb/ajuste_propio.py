"""Compuerta de cribado: TPE de 20 evaluaciones para una variante propia en la partición de
diseño, y evaluación de la mejor configuración en la partición disjunta.

Uso: python ajuste_propio.py <variante>
"""
import sys
import time
import numpy as np
import optuna
from comun import cargar, particiones, guardar_json, leer_json
import gbdt
from gbdt import BoostingNewton, Binarizador, xp

optuna.logging.set_verbosity(optuna.logging.WARNING)

VARIANTES = {
    "newton": {},                                   # #4
    "hojas_lineales": dict(hojas_lineales=True),    # #2
    "heterogeneo": dict(heterogeneo=True),          # #3
    "dart": dict(dart=True),                        # #7
    "olvidadizo": dict(olvidadizo=True),            # #6
    "mvs": dict(muestreo="mvs"),                    # #12
    "goss": dict(muestreo="goss"),                  # #26
}


def preparar(tr, va, lineal):
    X, y = cargar()
    Xtr, Xva = np.asarray(X[tr]), np.asarray(X[va])
    b = Binarizador().fit(Xtr)
    out = dict(Xb=xp.asarray(b.transform(Xtr)), Xbv=xp.asarray(b.transform(Xva)), y=y[tr], yv=y[va])
    if lineal:
        mu = np.nanmean(Xtr, 0)
        sd = np.nanstd(Xtr, 0) + 1e-6
        z = lambda A: xp.asarray(np.nan_to_num(np.clip((A - mu) / sd, -5, 5), nan=0.0).astype(np.float16))
        out["Z"], out["Zv"] = z(Xtr), z(Xva)
    return out


def params(t, var):
    p = dict(
        tasa=t.suggest_float("tasa", 0.01, 0.1, log=True),
        max_depth=t.suggest_int("max_depth", 3, 8),
        min_hess=t.suggest_float("min_hess", 1, 100, log=True),
        submuestra=t.suggest_float("submuestra", 0.5, 1.0),
        colsample=t.suggest_float("colsample", 0.2, 1.0),
        lam=t.suggest_float("lam", 1e-2, 100, log=True),
        alpha=t.suggest_float("alpha", 1e-3, 10, log=True),
        gamma=t.suggest_float("gamma", 1e-3, 5, log=True),
        peso_pos=t.suggest_float("peso_pos", 1, 5),
    )
    if var.get("hojas_lineales"):
        p["hojas_lineales"] = True
        p["lam_lin"] = t.suggest_float("lam_lin", 1, 1000, log=True)
    if var.get("heterogeneo"):
        p["prof_min"] = t.suggest_int("prof_min", 2, 4)
    if var.get("dart"):
        p["dart_tasa"] = t.suggest_float("dart_tasa", 0.005, 0.1, log=True)
    if var.get("olvidadizo"):
        p["olvidadizo"] = True
    if "muestreo" in var:
        p["muestreo"] = var["muestreo"]
    return p


def evaluar(p, D, n=5000):
    m = BoostingNewton(n_rondas=n, paciencia=100, eval_cada=10, **p)
    m.fit(D["Xb"], D["y"], D["Xbv"], D["yv"], Z=D.get("Z"), Z_val=D.get("Zv"))
    return m.mejor_auc, m.mejor_it


def desde_params(fp, var):
    """Reconstruye el dict de BoostingNewton a partir de los params de Optuna."""
    p = dict(fp)
    if var.get("hojas_lineales"):
        p["hojas_lineales"] = True
    if var.get("olvidadizo"):
        p["olvidadizo"] = True
    if "muestreo" in var:
        p["muestreo"] = var["muestreo"]
    return p


if __name__ == "__main__":
    nombre = sys.argv[1]
    var = VARIANTES[nombre]
    lineal = bool(var.get("hojas_lineales"))
    P = particiones()
    t0 = time.time()
    D = preparar(P["d_tr"], P["d_va"], lineal)
    estudio = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))

    def obj(t):
        p = params(t, var)
        a, it = evaluar(p, D)
        t.set_user_attr("it", it)
        print(f"[{nombre}] trial {t.number} auc {a:.5f} it {it} t={time.time() - t0:.0f}s", flush=True)
        return a

    estudio.optimize(obj, n_trials=20)
    mejor = estudio.best_trial
    p = desde_params(mejor.params, var)
    del D
    Dj = preparar(P["j_tr"], P["j_va"], lineal)
    a_j, it_j = evaluar(p, Dj)
    xgb = leer_json("xgb_ajuste.json")
    out = dict(variante=nombre, params=p, it=mejor.user_attrs["it"], auc_diseno=mejor.value,
               auc_disjunta=a_j, it_disjunta=it_j, segundos=time.time() - t0,
               xgb_diseno=xgb["auc_diseno"], xgb_disjunta=xgb["auc_disjunta"])
    out["pasa_compuerta"] = bool(out["auc_diseno"] > out["xgb_diseno"] and out["auc_disjunta"] > out["xgb_disjunta"])
    print(out, flush=True)
    guardar_json(f"compuerta_{nombre}.json", out)
