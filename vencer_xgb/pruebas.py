"""Las cuatro pruebas del protocolo: DeLong OOF, DeLong prueba, Nadeau–Bengio y Wilcoxon.

Uso: python pruebas.py <candidato> [<candidato2> ...]   (contra xgb; Holm por prueba)
"""
import os
import sys
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
from comun import cargar, particiones, guardar_json, RES


def _midrank(x):
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n)
    out[j] = t
    return out


def delong(y, a, b):
    """DeLong rápido (Sun y Xu 2014). Devuelve dif AUC (a−b), z, p bilateral."""
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    res = []
    for s in (a, b):
        tx, ty, tz = _midrank(s[pos]), _midrank(s[neg]), _midrank(np.concatenate([s[pos], s[neg]]))
        auc = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
        v01 = (tz[:m] - tx) / n
        v10 = 1 - (tz[m:] - ty) / m
        res.append((auc, v01, v10))
    sx = np.cov(np.vstack([res[0][1], res[1][1]]))
    sy = np.cov(np.vstack([res[0][2], res[1][2]]))
    S = sx / m + sy / n
    d = res[0][0] - res[1][0]
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = d / np.sqrt(var)
    return d, z, 2 * stats.norm.sf(abs(z))


def nadeau_bengio(dif, ratio=0.25):
    k = len(dif)
    var = np.var(dif, ddof=1)
    t = np.mean(dif) / np.sqrt((1 / k + ratio) * var)
    return t, 2 * stats.t.sf(abs(t), k - 1)


def holm(ps):
    ps = np.asarray(ps)
    orden = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    acum = 0
    for r, i in enumerate(orden):
        acum = max(acum, (m - r) * ps[i])
        adj[i] = min(1, acum)
    return adj


def comparar(cand, base="xgb"):
    _, y = cargar()
    P = particiones()
    ytr, yte = y[P["tr"]], y[P["te"]]
    A = np.load(os.path.join(RES, f"pred_{cand}.npz"))
    B = np.load(os.path.join(RES, f"pred_{base}.npz"))
    d_oof, _, p_oof = delong(ytr, A["oof0"], B["oof0"])
    d_te, _, p_te = delong(yte, A["test"], B["test"])
    dif = A["auc_pliegues"] - B["auc_pliegues"]
    t_nb, p_nb = nadeau_bengio(dif)
    p_w = stats.wilcoxon(dif).pvalue
    return dict(candidato=cand, auc_oof=roc_auc_score(ytr, A["oof0"]), auc_oof_xgb=roc_auc_score(ytr, B["oof0"]),
                auc_test=roc_auc_score(yte, A["test"]), auc_test_xgb=roc_auc_score(yte, B["test"]),
                dif_oof=d_oof, p_delong_oof=p_oof, dif_test=d_te, p_delong_test=p_te,
                dif_pliegues_media=float(dif.mean()), t_nb=t_nb, p_nb=p_nb, p_wilcoxon=p_w,
                pliegues_ganados=int((dif > 0).sum()))


if __name__ == "__main__":
    cands = sys.argv[1:]
    R = [comparar(c) for c in cands]
    for clave in ("p_delong_oof", "p_delong_test", "p_nb", "p_wilcoxon"):
        adj = holm([r[clave] for r in R])
        for r, a in zip(R, adj):
            r[clave + "_holm"] = a
    for r in R:
        r["veredicto"] = "SUPERA" if (r["dif_oof"] > 0 and r["dif_test"] > 0 and r["dif_pliegues_media"] > 0 and
                                      all(r[k + "_holm"] < 0.05 for k in ("p_delong_oof", "p_delong_test", "p_nb", "p_wilcoxon"))) else "NO SUPERA"
        print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
    guardar_json("pruebas_" + "_".join(cands) + ".json", R)
