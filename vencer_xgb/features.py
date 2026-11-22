"""Construcción de variables para Home Credit (igualdad de condiciones).

Todas las comparaciones de este directorio (XGBoost y los algoritmos propios)
usan exactamente la misma matriz que produce este archivo.
Salida: vencer_xgb/cache/X.npy (float32), y.npy, columnas.json
"""
import json
import os
import sys
import numpy as np
import pandas as pd

# Busca primero los datos dentro del proyecto (datos/cache, como vienen en el .zip);
# si no están, usa la carpeta original del equipo. También se puede fijar HOME_CREDIT_DATOS.
_AQUI = os.path.dirname(os.path.abspath(__file__))
_CANDIDATAS = [
    os.environ.get("HOME_CREDIT_DATOS", ""),
    os.path.join(_AQUI, "..", "datos", "cache"),
    r"C:\Users\samit\OneDrive\Escritorio\UNIVERSIDAD\2026\2026-3\MACHINE LEARNING\03_Proyecto_Home_Credit\datos\cache",
]
RAIZ_DATOS = next((c for c in _CANDIDATAS if c and os.path.exists(os.path.join(c, "application_train.parquet"))), _CANDIDATAS[1])
SALIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def leer(nombre):
    d = pd.read_parquet(os.path.join(RAIZ_DATOS, f"{nombre}.parquet"))
    for c in ("SK_ID_CURR", "SK_ID_PREV", "SK_ID_BUREAU"):
        if c in d.columns:
            d[c] = d[c].astype("int64")
    return d


def one_hot(d, nan_como_categoria=True):
    cat = [c for c in d.columns if d[c].dtype == "object" or str(d[c].dtype) == "category"]
    d = pd.get_dummies(d, columns=cat, dummy_na=nan_como_categoria, dtype=np.float32)
    nuevas = [c for c in d.columns if any(c.startswith(k + "_") for k in cat)]
    return d, nuevas


def agregar(d, llave, spec, prefijo):
    a = d.groupby(llave).agg(spec)
    a.columns = [f"{prefijo}_{c}_{f.upper()}" for c, f in a.columns]
    return a


def aplicacion():
    df = leer("application_train")
    df = df[df["CODE_GENDER"] != "XNA"]
    for c in ("CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY"):
        df[c], _ = pd.factorize(df[c])
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
    df["DAYS_EMPLOYED_PERC"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
    df["INCOME_CREDIT_PERC"] = df["AMT_INCOME_TOTAL"] / df["AMT_CREDIT"]
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]
    df["ANNUITY_INCOME_PERC"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["PAYMENT_RATE"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]
    df["CREDIT_GOODS_RATIO"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]
    ext = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    df["EXT_MEAN"] = ext.mean(axis=1)
    df["EXT_STD"] = ext.std(axis=1)
    df["EXT_PROD"] = ext.prod(axis=1, min_count=3)
    df["EXT_MIN"] = ext.min(axis=1)
    df["EXT_MAX"] = ext.max(axis=1)
    df["EXT_NAN"] = ext.isna().sum(axis=1)
    df, _ = one_hot(df, False)
    return df


def buro():
    b = leer("bureau")
    bb = leer("bureau_balance")
    bb, bb_cat = one_hot(bb)
    spec = {"MONTHS_BALANCE": ["min", "max", "size"]}
    spec.update({c: ["mean"] for c in bb_cat})
    bba = bb.groupby("SK_ID_BUREAU").agg(spec)
    bba.columns = [f"{c}_{f.upper()}" for c, f in bba.columns]
    b = b.join(bba, on="SK_ID_BUREAU")
    b, b_cat = one_hot(b)
    num = {
        "DAYS_CREDIT": ["min", "max", "mean", "var"],
        "DAYS_CREDIT_ENDDATE": ["min", "max", "mean"],
        "DAYS_CREDIT_UPDATE": ["mean"],
        "CREDIT_DAY_OVERDUE": ["max", "mean"],
        "AMT_CREDIT_MAX_OVERDUE": ["mean"],
        "AMT_CREDIT_SUM": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_DEBT": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_OVERDUE": ["mean"],
        "AMT_CREDIT_SUM_LIMIT": ["mean", "sum"],
        "AMT_ANNUITY": ["max", "mean"],
        "CNT_CREDIT_PROLONG": ["sum"],
        "MONTHS_BALANCE_MIN": ["min"],
        "MONTHS_BALANCE_MAX": ["max"],
        "MONTHS_BALANCE_SIZE": ["mean", "sum"],
    }
    cat = {c: ["mean"] for c in b_cat + [f"{c}_MEAN" for c in bb_cat]}
    out = agregar(b, "SK_ID_CURR", {**num, **cat}, "BURO")
    act = agregar(b[b["CREDIT_ACTIVE_Active"] == 1], "SK_ID_CURR", num, "ACTIVE")
    cer = agregar(b[b["CREDIT_ACTIVE_Closed"] == 1], "SK_ID_CURR", num, "CLOSED")
    out = out.join(act).join(cer)
    out["BURO_DEBT_CREDIT_RATIO"] = out["BURO_AMT_CREDIT_SUM_DEBT_SUM"] / out["BURO_AMT_CREDIT_SUM_SUM"]
    return out


def previas():
    p = leer("previous_application")
    p, cat = one_hot(p)
    for c in ("DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION", "DAYS_LAST_DUE", "DAYS_TERMINATION"):
        p[c] = p[c].replace(365243, np.nan)
    p["APP_CREDIT_PERC"] = p["AMT_APPLICATION"] / p["AMT_CREDIT"]
    num = {
        "AMT_ANNUITY": ["min", "max", "mean"],
        "AMT_APPLICATION": ["min", "max", "mean"],
        "AMT_CREDIT": ["min", "max", "mean"],
        "APP_CREDIT_PERC": ["min", "max", "mean", "var"],
        "AMT_DOWN_PAYMENT": ["min", "max", "mean"],
        "AMT_GOODS_PRICE": ["min", "max", "mean"],
        "HOUR_APPR_PROCESS_START": ["min", "max", "mean"],
        "RATE_DOWN_PAYMENT": ["min", "max", "mean"],
        "DAYS_DECISION": ["min", "max", "mean"],
        "CNT_PAYMENT": ["mean", "sum"],
    }
    spec = {**num, **{c: ["mean"] for c in cat}}
    out = agregar(p, "SK_ID_CURR", spec, "PREV")
    ap = agregar(p[p["NAME_CONTRACT_STATUS_Approved"] == 1], "SK_ID_CURR", num, "APPROVED")
    re = agregar(p[p["NAME_CONTRACT_STATUS_Refused"] == 1], "SK_ID_CURR", num, "REFUSED")
    return out.join(ap).join(re)


def pos():
    d = leer("POS_CASH_balance")
    d, cat = one_hot(d)
    spec = {"MONTHS_BALANCE": ["max", "mean", "size"], "SK_DPD": ["max", "mean"], "SK_DPD_DEF": ["max", "mean"]}
    spec.update({c: ["mean"] for c in cat})
    out = agregar(d, "SK_ID_CURR", spec, "POS")
    out["POS_COUNT"] = d.groupby("SK_ID_CURR").size()
    return out


def cuotas():
    d = leer("installments_payments")
    d["PAYMENT_PERC"] = d["AMT_PAYMENT"] / d["AMT_INSTALMENT"]
    d["PAYMENT_DIFF"] = d["AMT_INSTALMENT"] - d["AMT_PAYMENT"]
    d["DPD"] = (d["DAYS_ENTRY_PAYMENT"] - d["DAYS_INSTALMENT"]).clip(lower=0)
    d["DBD"] = (d["DAYS_INSTALMENT"] - d["DAYS_ENTRY_PAYMENT"]).clip(lower=0)
    spec = {
        "NUM_INSTALMENT_VERSION": ["nunique"],
        "DPD": ["max", "mean", "sum"],
        "DBD": ["max", "mean", "sum"],
        "PAYMENT_PERC": ["max", "mean", "sum", "var"],
        "PAYMENT_DIFF": ["max", "mean", "sum", "var"],
        "AMT_INSTALMENT": ["max", "mean", "sum"],
        "AMT_PAYMENT": ["min", "max", "mean", "sum"],
        "DAYS_ENTRY_PAYMENT": ["max", "mean", "sum"],
    }
    out = agregar(d, "SK_ID_CURR", spec, "INSTAL")
    out["INSTAL_COUNT"] = d.groupby("SK_ID_CURR").size()
    rec = d[d["DAYS_INSTALMENT"] > -365]
    out = out.join(agregar(rec, "SK_ID_CURR", {"DPD": ["max", "mean"], "PAYMENT_DIFF": ["mean", "sum"], "PAYMENT_PERC": ["mean"]}, "INSTAL12M"))
    return out


def tarjeta():
    d = leer("credit_card_balance")
    d, _ = one_hot(d)
    d = d.drop(columns=["SK_ID_PREV"])
    d["UTIL"] = d["AMT_BALANCE"] / d["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)
    cols = [c for c in d.columns if c != "SK_ID_CURR"]
    out = agregar(d, "SK_ID_CURR", {c: ["min", "max", "mean", "sum", "var"] for c in cols}, "CC")
    out["CC_COUNT"] = d.groupby("SK_ID_CURR").size()
    return out


def construir():
    os.makedirs(SALIDA, exist_ok=True)
    df = aplicacion().set_index("SK_ID_CURR")
    for nombre, f in (("buro", buro), ("previas", previas), ("pos", pos), ("cuotas", cuotas), ("tarjeta", tarjeta)):
        t = f()
        print(nombre, t.shape, flush=True)
        df = df.join(t)
        for presencia in (nombre,):
            df[f"TIENE_{presencia.upper()}"] = df.index.isin(t.index).astype(np.float32)
    y = df.pop("TARGET").to_numpy(np.int8)
    df = df.replace([np.inf, -np.inf], np.nan)
    # quitar columnas constantes o casi vacías (criterio fijo, sin mirar y)
    nun = df.nunique(dropna=True)
    df = df.loc[:, nun > 1]
    df = df.loc[:, df.isna().mean() < 0.999]
    X = df.to_numpy(np.float32)
    np.save(os.path.join(SALIDA, "X.npy"), X)
    np.save(os.path.join(SALIDA, "y.npy"), y)
    np.save(os.path.join(SALIDA, "ids.npy"), df.index.to_numpy())
    json.dump(list(df.columns), open(os.path.join(SALIDA, "columnas.json"), "w"))
    print("X", X.shape, "positivos", y.mean())


if __name__ == "__main__":
    construir()
