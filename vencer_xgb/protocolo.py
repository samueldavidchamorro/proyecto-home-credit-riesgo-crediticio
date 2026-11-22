"""Protocolo completo: VC 5 pliegues × 2 repeticiones sobre el 80 % + prueba 20 %.

Uso: python protocolo.py xgb
     python protocolo.py <variante>        (lee resultados/compuerta_<variante>.json)
Guarda resultados/pred_<nombre>.npz con oof0, oof1, test, auc_pliegues (10,).
"""
import os
import sys
import time
import numpy as np
from sklearn.metrics import roc_auc_score
from comun import cargar, particiones, pliegues, leer_json, RES

MULT_RONDAS = 1.2


def correr_xgb(Xtr, ytr, Xva, Xte, info):
    import xgboost as xgb
    p = dict(objective="binary:logistic", tree_method="hist", device="cuda", seed=0, **info["params"])
    b = xgb.train(p, xgb.DMatrix(Xtr, label=ytr), int(info["it"] * MULT_RONDAS))
    return b.predict(xgb.DMatrix(Xva), output_margin=True), (b.predict(xgb.DMatrix(Xte), output_margin=True) if Xte is not None else None)


def correr_propio(Xtr, ytr, Xva, Xte, info):
    from gbdt import BoostingNewton, Binarizador, xp, a_np
    p = dict(info["params"])
    bz = Binarizador().fit(Xtr)
    Z = Zv = Zt = None
    if p.get("hojas_lineales"):
        mu = np.nanmean(Xtr, 0)
        sd = np.nanstd(Xtr, 0) + 1e-6
        z = lambda A: xp.asarray(np.nan_to_num(np.clip((A - mu) / sd, -5, 5), nan=0.0).astype(np.float16))
        Z, Zv = z(Xtr), z(Xva)
        Zt = z(Xte) if Xte is not None else None
    m = BoostingNewton(n_rondas=int(info["it"] * MULT_RONDAS), **p)
    m.fit(xp.asarray(bz.transform(Xtr)), ytr, Z=Z)
    pv = a_np(m.decision(bz.transform(Xva), Zv))
    pt = a_np(m.decision(bz.transform(Xte), Zt)) if Xte is not None else None
    return pv, pt


def ejecutar(nombre):
    X, y = cargar()
    P = particiones()
    tr, te = P["tr"], P["te"]
    ytr = y[tr]
    Xte = np.asarray(X[te])
    if nombre == "xgb":
        info, f = leer_json("xgb_ajuste.json"), correr_xgb
    else:
        base, _, sem = nombre.partition("_s")
        if sem.isdigit():  # réplica por semilla (#11): mismos hiperparámetros, otra semilla
            info = leer_json(f"compuerta_{base}.json")
            info["params"] = dict(info["params"], semilla=int(sem))
        else:
            info = leer_json(f"compuerta_{nombre}.json")
        f = correr_propio
    ruta = os.path.join(RES, f"pred_{nombre}.npz")
    parcial = dict(np.load(ruta)) if os.path.exists(ruta) else {}
    oof = [parcial.get("oof0", np.full(len(tr), np.nan)), parcial.get("oof1", np.full(len(tr), np.nan))]
    test = parcial.get("test", np.zeros(len(te)))
    aucs = parcial.get("auc_pliegues", np.full(10, np.nan))
    t0 = time.time()
    for rep in (0, 1):
        for k, (a, b) in enumerate(pliegues(ytr, rep)):
            j = rep * 5 + k
            if not np.isnan(aucs[j]):
                continue
            pv, pt = f(np.asarray(X[tr[a]]), ytr[a], np.asarray(X[tr[b]]), Xte if rep == 0 else None, info)
            oof[rep][b] = pv
            if rep == 0:
                test = test + pt / 5
            aucs[j] = roc_auc_score(ytr[b], pv)
            print(f"[{nombre}] rep {rep} pliegue {k} auc {aucs[j]:.5f} t={time.time() - t0:.0f}s", flush=True)
            np.savez(ruta, oof0=oof[0], oof1=oof[1], test=test, auc_pliegues=aucs)
    print(f"[{nombre}] OOF0 {roc_auc_score(ytr, oof[0]):.5f}  prueba {roc_auc_score(y[te], test):.5f}  media pliegues {np.mean(aucs):.5f}")


if __name__ == "__main__":
    ejecutar(sys.argv[1])
