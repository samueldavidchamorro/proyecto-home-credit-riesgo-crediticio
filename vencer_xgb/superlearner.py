"""#1 Super Learner con biblioteca de boostings propios y selección de Caruana (pesos en el símplex).

Uso: python superlearner.py <nombre_salida> <miembro1> <miembro2> ...
Los miembros son modelos propios con pred_<m>.npz ya calculado por protocolo.py.
Los pesos del pliegue k (repetición r) se estiman solo con las filas OOF fuera de ese pliegue;
los pesos para prueba se estiman con todo el OOF de la repetición 0. XGBoost nunca entra.
"""
import os
import sys
import numpy as np
from sklearn.metrics import roc_auc_score
from comun import cargar, particiones, pliegues, guardar_json, RES


def caruana(M, y, rondas=50):
    """Selección con reemplazo (Caruana et al. 2004) maximizando AUC. M: (n, k) márgenes estandarizados."""
    k = M.shape[1]
    cuenta = np.zeros(k)
    actual = np.zeros(len(y))
    for r in range(rondas):
        mejor, j_mejor = -1, 0
        for j in range(k):
            a = roc_auc_score(y, (actual * r + M[:, j]) / (r + 1))
            if a > mejor:
                mejor, j_mejor = a, j
        cuenta[j_mejor] += 1
        actual = (actual * r + M[:, j_mejor]) / (r + 1)
    return cuenta / cuenta.sum()


def estandarizar(M, ref):
    return (M - ref.mean(0)) / ref.std(0)


if __name__ == "__main__":
    salida, miembros = sys.argv[1], sys.argv[2:]
    _, y = cargar()
    P = particiones()
    ytr = y[P["tr"]]
    D = [np.load(os.path.join(RES, f"pred_{m}.npz")) for m in miembros]
    oof = [np.column_stack([d[f"oof{r}"] for d in D]) for r in (0, 1)]
    T = np.column_stack([d["test"] for d in D])
    sl_oof = [np.zeros(len(ytr)), np.zeros(len(ytr))]
    aucs = np.zeros(10)
    pesos_pl = []
    for rep in (0, 1):
        M = estandarizar(oof[rep], oof[rep])
        for k, (_, b) in enumerate(pliegues(ytr, rep)):
            fuera = np.setdiff1d(np.arange(len(ytr)), b)
            w = caruana(M[fuera], ytr[fuera])
            pesos_pl.append(w.tolist())
            sl_oof[rep][b] = M[b] @ w
            aucs[rep * 5 + k] = roc_auc_score(ytr[b], sl_oof[rep][b])
    w_test = caruana(estandarizar(oof[0], oof[0]), ytr)
    test = estandarizar(T, oof[0]) @ w_test
    np.savez(os.path.join(RES, f"pred_{salida}.npz"), oof0=sl_oof[0], oof1=sl_oof[1], test=test, auc_pliegues=aucs)
    info = dict(miembros=miembros, pesos_test=dict(zip(miembros, w_test.tolist())), pesos_pliegues=pesos_pl,
                auc_oof0=roc_auc_score(ytr, sl_oof[0]), auc_test=roc_auc_score(y[P["te"]], test),
                auc_miembros_oof0={m: roc_auc_score(ytr, d["oof0"]) for m, d in zip(miembros, D)})
    print(info["pesos_test"], info["auc_oof0"], info["auc_test"], info["auc_miembros_oof0"])
    guardar_json(f"superlearner_{salida}.json", info)
