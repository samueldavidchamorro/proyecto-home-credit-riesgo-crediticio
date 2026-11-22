"""Particiones fijadas por el protocolo y utilidades compartidas."""
import json
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

DIR = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(DIR, "cache")
RES = os.path.join(DIR, "resultados")
os.makedirs(RES, exist_ok=True)


def cargar():
    X = np.load(os.path.join(CACHE, "X.npy"), mmap_mode="r")
    y = np.load(os.path.join(CACHE, "y.npy"))
    return X, y


def particiones():
    """Devuelve índices: entrenamiento (80 %), prueba (20 %), diseño y disjunta."""
    ruta = os.path.join(CACHE, "particiones.npz")
    if os.path.exists(ruta):
        d = np.load(ruta)
        return {k: d[k] for k in d.files}
    _, y = cargar()
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=2026)
    tr = np.sort(tr)
    te = np.sort(te)
    # diseño: 64k / 16k del entrenamiento; disjunta: otras 64k / 16k sin solapamiento
    resto, d_all = train_test_split(tr, test_size=80_000, stratify=y[tr], random_state=11)
    d_tr, d_va = train_test_split(d_all, test_size=16_000, stratify=y[d_all], random_state=11)
    _, j_all = train_test_split(resto, test_size=80_000, stratify=y[resto], random_state=12)
    j_tr, j_va = train_test_split(j_all, test_size=16_000, stratify=y[j_all], random_state=12)
    out = dict(tr=tr, te=te, d_tr=np.sort(d_tr), d_va=np.sort(d_va), j_tr=np.sort(j_tr), j_va=np.sort(j_va))
    assert len(np.intersect1d(d_all, j_all)) == 0
    np.savez(ruta, **out)
    return out


def pliegues(y_tr, repeticion):
    skf = StratifiedKFold(5, shuffle=True, random_state=repeticion)
    return list(skf.split(np.zeros(len(y_tr)), y_tr))


def guardar_json(nombre, obj):
    with open(os.path.join(RES, nombre), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=float)


def leer_json(nombre):
    with open(os.path.join(RES, nombre), encoding="utf-8") as f:
        return json.load(f)
