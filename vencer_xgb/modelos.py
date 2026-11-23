"""Los 21 algoritmos restantes del Top 30, con código propio (API de NumPy; en GPU con CuPy).

Interfaz común:  m = Clase(**params); m.fit(X, y, Xv=None, yv=None); m.decision(X) -> márgenes (numpy)
X son las variables crudas (float32 con NaN) de `features.py`. Tras fit con validación quedan
m.mejor_auc (AUC en validación) y m.mejor_it (rondas o niveles elegidos; None si no aplica).

CATALOGO[nombre] = (constructor, espacio TPE, tipo de `it`)  — ver PROTOCOLO.md, sección «Ampliación».
No usa xgboost, lightgbm, catboost, torch, tensorflow ni sklearn.ensemble.
"""
import math
import numpy as np
import gbdt
from gbdt import BoostingNewton, Binarizador, xp, a_np, auc, suma_dispersa, NAN_BIN, Arbol

if gbdt.GPU:
    from cupyx.scipy.special import ndtr, ndtri
else:  # pragma: no cover
    from scipy.special import ndtr, ndtri


# ====================================================================== preparación compartida
class Prep:
    """Binarizador, estandarización y caché de transformaciones del conjunto de entrenamiento actual."""

    def __init__(self):
        self.tr = None

    def ajustar(self, X):
        if self.tr is X:
            return self
        self.tr = X
        self.bz = Binarizador().fit(X)
        self.mu = np.nanmean(X, 0).astype(np.float32)
        self.sd = (np.nanstd(X, 0) + 1e-6).astype(np.float32)
        S = X[np.random.default_rng(0).choice(len(X), min(len(X), 20000), replace=False)]
        nun = np.array([len(np.unique(c[~np.isnan(c)])) for c in S.T])
        self.cont = np.nonzero(nun > 2)[0]          # variables continuas (para rotar / proyectar / RFF)
        self.nocont = np.nonzero(nun <= 2)[0]
        self.cache = {}
        return self

    def _memo(self, clave, X, f):
        k = (clave, id(X))
        if k not in self.cache or self.cache[k][0] is not X:
            if len(self.cache) > 12:
                self.cache = {kk: v for kk, v in self.cache.items() if v[0] is self.tr}
            self.cache[k] = (X, f(X))
        return self.cache[k][1]

    def xb(self, X):
        return self._memo("xb", X, lambda A: xp.asarray(self.bz.transform(A)))

    def z(self, X):
        """Estandarizada, faltantes a 0 (la media), recortada a ±5, float16 en GPU."""
        return self._memo("z", X, lambda A: xp.asarray(
            np.nan_to_num(np.clip((A - self.mu) / self.sd, -5, 5), nan=0.0).astype(np.float16)))


PREP = Prep()


def umbrales_gpu(A, max_bins=255):
    """Bordes por cuantiles de cada columna de A (GPU, sin NaN)."""
    n = A.shape[0]
    S = A if n <= 100_000 else A[xp.asarray(np.random.default_rng(0).choice(n, 100_000, replace=False))]
    q = xp.quantile(S.astype(xp.float32), xp.linspace(0, 1, max_bins, dtype=xp.float32)[1:-1], axis=0)  # (253, c)
    return q.T.copy()  # (c, 253)


def binarizar_gpu(A, bordes):
    out = xp.empty(A.shape, dtype=xp.uint8)
    for j in range(A.shape[1]):
        out[:, j] = xp.searchsorted(bordes[j], A[:, j].astype(xp.float32)).astype(xp.uint8)
    return out


def _sig(F):
    return 1 / (1 + xp.exp(-F))


def _logit(p):
    p = xp.clip(p, 1e-6, 1 - 1e-6)
    return xp.log(p / (1 - p))


def _auc_np(y, s):
    return auc(xp.asarray(y), xp.asarray(s))


# ====================================================================== boostings (variantes del motor)
class Booster:
    """Envoltura de BoostingNewton para las variantes de boosting de la ampliación.

    vistas="rotacion" (#8): cada vista = variables originales + bloques rotados al azar de `vista_m` continuas.
    vistas="oblicuo"  (#9): cada vista = originales + `vista_m` proyecciones dispersas ±1 de `densidad` continuas.
    nucleo_D > 0      (#5): rasgos aleatorios de Fourier (núcleo RBF, escala `nucleo_ls`) para el aprendiz de núcleo.
    El resto de parámetros pasa directo a BoostingNewton.
    """
    tipo_it = "rondas"

    def __init__(self, vistas=None, n_vistas=3, vista_m=64, bloque=4, densidad=3, nucleo_D=0, nucleo_ls=20.0,
                 **p):
        self.vistas, self.n_vistas, self.vista_m, self.bloque, self.densidad = vistas, n_vistas, vista_m, bloque, densidad
        self.nucleo_D, self.nucleo_ls = nucleo_D, nucleo_ls
        self.p = p

    def _proyecciones(self, rng):
        C = len(PREP.cont)
        Ps = []
        for _ in range(self.n_vistas):
            P = np.zeros((C, self.vista_m), dtype=np.float32)
            if self.vistas == "rotacion":
                cols = rng.choice(C, min(C, self.vista_m), replace=False)
                for a in range(0, len(cols), self.bloque):
                    b = cols[a:a + self.bloque]
                    Q, _ = np.linalg.qr(rng.standard_normal((len(b), len(b))))
                    P[np.ix_(b, np.arange(a, a + len(b)))] = Q
            else:
                for k in range(self.vista_m):
                    b = rng.choice(C, self.densidad, replace=False)
                    P[b, k] = rng.choice([-1.0, 1.0], self.densidad)
            Ps.append(xp.asarray(P))
        return Ps

    def _extra(self, X):
        """Vistas (lista de matrices de cubetas) y rasgos de Fourier para X."""
        Xb = PREP.xb(X)
        V, Phi = Xb, None
        if self.vistas:
            Zc = PREP.z(X)[:, xp.asarray(PREP.cont)].astype(xp.float32)
            V = []
            for P, bordes in zip(self.P, self.bordes):
                E = Zc @ P
                V.append(xp.concatenate([Xb, binarizar_gpu(E, bordes)], axis=1))
            del Zc
        if self.nucleo_D:
            Zf = PREP.z(X).astype(xp.float32)
            Phi = math.sqrt(2.0 / self.nucleo_D) * xp.cos(Zf @ self.W + self.b)
        return V, Phi

    def fit(self, X, y, Xv=None, yv=None):
        PREP.ajustar(X)
        rng = np.random.default_rng(self.p.get("semilla", 0) + 1000)
        if self.vistas:
            self.P = self._proyecciones(rng)
            Zc = PREP.z(X)[:, xp.asarray(PREP.cont)].astype(xp.float32)
            self.bordes = [umbrales_gpu(Zc @ P) for P in self.P]
            del Zc
        if self.nucleo_D:
            F = X.shape[1]
            self.W = xp.asarray(rng.standard_normal((F, self.nucleo_D)).astype(np.float32) / self.nucleo_ls)
            self.b = xp.asarray(rng.uniform(0, 2 * np.pi, self.nucleo_D).astype(np.float32))
        V, Phi = self._extra(X)
        Vv, Phiv = self._extra(Xv) if Xv is not None else (None, None)
        Z = Zv = None
        if self.p.get("hojas_lineales"):
            Z, Zv = PREP.z(X), (PREP.z(Xv) if Xv is not None else None)
        self.m = BoostingNewton(**self.p)
        self.m.fit(V, y, Vv, yv, Z=Z, Z_val=Zv, Phi=Phi, Phi_val=Phiv)
        self.mejor_auc, self.mejor_it = self.m.mejor_auc, self.m.mejor_it
        return self

    def decision(self, X):
        V, Phi = self._extra(X)
        Z = PREP.z(X) if self.p.get("hojas_lineales") else None
        return a_np(self.m.decision(V, Z, Phi))


# ====================================================================== bosques (#20, #21, #22)
class Bosque:
    """Bosque de árboles de regresión sobre y con el motor de histogramas (equivale a Gini/varianza).

    aleatorio=True: árboles extremadamente aleatorios (#22, un umbral al azar por nodo y variable).
    rotacion=True : bosque de rotación (#21, PCA de subconjuntos aleatorios de continuas en una muestra bootstrap).
    Bootstrap por pesos de Poisson(1) (salvo extra trees, que usan todas las filas, como Geurts et al.).
    """
    tipo_it = None

    def __init__(self, n_arboles=100, prof=8, col_arbol=0.3, col_nodo=0.5, min_hoja=5.0, aleatorio=False,
                 rotacion=False, rot_k=5, semilla=0):
        self.__dict__.update({k: v for k, v in locals().items() if k != "self"})

    def _constructor(self):
        return BoostingNewton(tasa=1.0, max_depth=self.prof, lam=1e-3, alpha=0.0, gamma=0.0, min_hess=self.min_hoja,
                              colsample=self.col_arbol, col_nodo=self.col_nodo, corte_aleatorio=self.aleatorio,
                              semilla=self.semilla)

    def _rotacion(self, Zc, rng, n):
        C = Zc.shape[1]
        perm = rng.permutation(C)
        R = xp.zeros((C, C), dtype=xp.float32)
        for a in range(0, C, self.rot_k):
            b = perm[a:a + self.rot_k]
            filas = xp.asarray(rng.choice(n, max(len(b) + 1, int(0.75 * n)), replace=True))
            B = Zc[filas][:, xp.asarray(b)]
            B = B - B.mean(0)
            _, _, Vt = xp.linalg.svd(B, full_matrices=False)
            bi = xp.asarray(b)
            R[bi[:, None], bi[None, :]] = Vt.T
        return R

    def fit_binario(self, Xb, y, Zc=None):
        """Ajuste sobre cubetas ya calculadas (lo usa el bosque profundo)."""
        y = xp.asarray(y).astype(xp.float32)
        n = Xb.shape[0]
        rng = np.random.default_rng(self.semilla)
        rg = xp.random.default_rng(self.semilla) if gbdt.GPU else rng
        self.arboles, self.rot = [], []
        cons = self._constructor()
        cons._rng_gpu = rg
        for t in range(self.n_arboles):
            w = xp.ones(n, xp.float32) if self.aleatorio else rg.poisson(1.0, n).astype(xp.float32)
            Xt = Xb
            if self.rotacion:
                R = self._rotacion(Zc, rng, n)
                E = Zc @ R
                bordes = umbrales_gpu(E)
                Xt = xp.concatenate([Xb[:, xp.asarray(PREP.nocont)], binarizar_gpu(E, bordes)], axis=1)
                self.rot.append((R, bordes))
                del E
            idx = xp.nonzero(w > 0)[0]
            arbol, _ = cons._crecer(Xt, -y * w, w, idx, self.prof, rng, None)
            # hoja = media de y (ponderada) en la hoja
            m_ = len(arbol.hoja)
            pos = arbol.rutear(Xt)
            sy = suma_dispersa(m_, pos, y * w)
            sw = suma_dispersa(m_, pos, w)
            arbol.valor = (sy / xp.maximum(sw, 1e-9)).astype(xp.float32)
            if hasattr(arbol, "_G"):
                del arbol._G, arbol._H
            self.arboles.append(arbol)
        return self

    def prob_binario(self, Xb, Zc=None):
        P = xp.zeros(Xb.shape[0], xp.float32)
        for t, a in enumerate(self.arboles):
            Xt = Xb
            if self.rotacion:
                R, bordes = self.rot[t]
                Xt = xp.concatenate([Xb[:, xp.asarray(PREP.nocont)], binarizar_gpu(Zc @ R, bordes)], axis=1)
            P += a.valor[a.rutear(Xt)]
        return P / len(self.arboles)

    def _zc(self, X):
        return PREP.z(X)[:, xp.asarray(PREP.cont)].astype(xp.float32) if self.rotacion else None

    def fit(self, X, y, Xv=None, yv=None):
        PREP.ajustar(X)
        self.fit_binario(PREP.xb(X), y, self._zc(X))
        self.mejor_it = None
        self.mejor_auc = _auc_np(yv, self.decision(Xv)) if Xv is not None else None
        return self

    def decision(self, X):
        return a_np(_logit(self.prob_binario(PREP.xb(X), self._zc(X))))


class BosqueProfundo:
    """#20 Bosque profundo en cascada (Zhou y Feng 2017): por nivel, un bosque aleatorio y uno extremadamente
    aleatorio; sus probabilidades fuera de pliegue (3 pliegues) se añaden como variables al nivel siguiente.
    El número de niveles se elige por la AUC de validación (it = niveles)."""
    tipo_it = "niveles"

    def __init__(self, n_arboles=50, prof=8, col_arbol=0.3, col_nodo=0.5, min_hoja=5.0, niveles_max=3, n_rondas=None,
                 semilla=0, **_):
        self.__dict__.update({k: v for k, v in locals().items() if k not in ("self", "_")})

    def _bosques(self, s):
        kw = dict(n_arboles=self.n_arboles, prof=self.prof, col_arbol=self.col_arbol, col_nodo=self.col_nodo,
                  min_hoja=self.min_hoja)
        return [Bosque(aleatorio=False, semilla=s, **kw), Bosque(aleatorio=True, semilla=s + 1, **kw)]

    def fit(self, X, y, Xv=None, yv=None):
        PREP.ajustar(X)
        Xb = PREP.xb(X)
        yg = xp.asarray(y).astype(xp.float32)
        n = len(y)
        niveles = self.n_rondas or self.niveles_max
        rng = np.random.default_rng(self.semilla)
        pl = rng.integers(0, 3, n)
        self.niveles = []
        Xl = Xb
        Xlv = PREP.xb(Xv) if Xv is not None else None
        mejor, mejor_n = -1, 0
        for lv in range(niveles):
            cv = xp.zeros((n, 2), xp.float32)
            modelos = []
            for k in range(3):
                a = xp.asarray(np.nonzero(pl != k)[0])
                b = xp.asarray(np.nonzero(pl == k)[0])
                ms = [f.fit_binario(Xl[a], yg[a]) for f in self._bosques(self.semilla + 10 * lv + 2 * k)]
                for j, f in enumerate(ms):
                    cv[b, j] = f.prob_binario(Xl[b])
                modelos.append(ms)
            bordes = umbrales_gpu(cv)
            self.niveles.append((modelos, bordes))
            Xl = xp.concatenate([Xb, binarizar_gpu(cv, bordes)], axis=1)
            if Xv is not None:
                pv = self._prob_nivel(Xlv, modelos)
                a_v = _auc_np(yv, pv.mean(1))
                if a_v > mejor:
                    mejor, mejor_n = a_v, lv + 1
                elif self.n_rondas is None:
                    break
                Xlv = xp.concatenate([PREP.xb(Xv), binarizar_gpu(pv, bordes)], axis=1)
        if Xv is not None and self.n_rondas is None:
            self.niveles = self.niveles[:mejor_n]
        self.mejor_auc, self.mejor_it = (mejor, mejor_n) if Xv is not None else (None, len(self.niveles))
        return self

    @staticmethod
    def _prob_nivel(Xl, modelos):
        return xp.stack([xp.stack([f.prob_binario(Xl) for f in ms], 1) for ms in modelos], 0).mean(0)

    def decision(self, X):
        Xb = PREP.xb(X)
        Xl = Xb
        for i, (modelos, bordes) in enumerate(self.niveles):
            p = self._prob_nivel(Xl, modelos)
            if i == len(self.niveles) - 1:
                return a_np(_logit(p.mean(1)))
            Xl = xp.concatenate([Xb, binarizar_gpu(p, bordes)], axis=1)


# ====================================================================== #14 LCE
class LCE:
    """Ensamble en cascada local (Fauvel et al.): por bolsa bootstrap, un boosting en la raíz; el nodo se corta
    por la mejor partición de Newton sobre el gradiente de ese margen, y en cada hijo un boosting parte del margen
    del padre (lo continúa) y además lo recibe como variable. Se promedian los márgenes de las bolsas."""
    tipo_it = None

    def __init__(self, n_bolsas=4, prof_cascada=1, n_r=300, tasa=0.05, max_depth=5, colsample=0.5, lam=1.0,
                 min_hess=10.0, min_filas=2000, semilla=0):
        self.__dict__.update({k: v for k, v in locals().items() if k != "self"})

    def _boost(self, s):
        return BoostingNewton(n_rondas=self.n_r, tasa=self.tasa, max_depth=self.max_depth, colsample=self.colsample,
                              lam=self.lam, min_hess=self.min_hess, submuestra=0.8, semilla=s)

    def _nodo(self, Xb, y, F0, prof, s, rng):
        m = self._boost(s)
        if F0 is None:
            m.fit(Xb, y)
            F = m.decision(Xb)
            aum = False
        else:
            bord = umbrales_gpu(F0[:, None])
            Xa = xp.concatenate([Xb, binarizar_gpu(F0[:, None], bord)], axis=1)
            m.fit(Xa, y, F0=F0)
            F = m.decision(Xa, F0=F0)
            aum = bord
        nodo = dict(m=m, aum=aum, hijos=None)
        if prof > 0 and len(y) >= 2 * self.min_filas:
            p = _sig(F)
            g, h = p - y, xp.maximum(p * (1 - p), 1e-6)
            c = BoostingNewton(max_depth=1, lam=1.0, min_hess=float(self.min_filas) * 0.05, colsample=1.0)
            c._rng_gpu = None
            arb, pos = c._crecer(Xb, g, h, xp.arange(len(y)), 1, rng, None)
            if not bool(arb.hoja[0]):
                izq = arb.rutear(Xb) == 1
                hijos = []
                for lado in (izq, ~izq):
                    ii = xp.nonzero(lado)[0]
                    if len(ii) < self.min_filas:
                        hijos = None
                        break
                    hijos.append(self._nodo(Xb[ii], y[ii], F[ii], prof - 1, s + 1, rng))
                if hijos:
                    nodo.update(hijos=hijos, corte=arb)
        return nodo

    def _pred(self, nodo, Xb, F0):
        if nodo["aum"] is False:
            F = nodo["m"].decision(Xb)
        else:
            Xa = xp.concatenate([Xb, binarizar_gpu(F0[:, None], nodo["aum"])], axis=1)
            F = nodo["m"].decision(Xa, F0=F0)
        if nodo["hijos"] is None:
            return F
        izq = nodo["corte"].rutear(Xb) == 1
        out = F.copy()
        for lado, hijo in zip((izq, ~izq), nodo["hijos"]):
            ii = xp.nonzero(lado)[0]
            if len(ii):
                out[ii] = self._pred(hijo, Xb[ii], F[ii])
        return out

    def fit(self, X, y, Xv=None, yv=None):
        PREP.ajustar(X)
        Xb = PREP.xb(X)
        yg = xp.asarray(y).astype(xp.float32)
        rng = np.random.default_rng(self.semilla)
        self.bolsas = []
        for b in range(self.n_bolsas):
            ii = xp.asarray(np.sort(rng.choice(len(y), len(y), replace=True)))
            self.bolsas.append(self._nodo(Xb[ii], yg[ii], None, self.prof_cascada, self.semilla + 100 * b, rng))
        self.mejor_it = None
        self.mejor_auc = _auc_np(yv, self.decision(Xv)) if Xv is not None else None
        return self

    def decision(self, X):
        Xb = PREP.xb(X)
        return a_np(sum(self._pred(n, Xb, None) for n in self.bolsas) / len(self.bolsas))


# ====================================================================== #23 SPE
class SPE:
    """Self-paced ensemble (Liu et al. 2020): en cada iteración se submuestrea la clase mayoritaria por cubetas de
    dureza |p − y| del ensamble actual, con peso 1/(dureza media + α_i), α_i = tan(iπ/2n). Base = boosting propio."""
    tipo_it = None

    def __init__(self, n_est=10, k_cubetas=10, n_r=300, tasa=0.05, max_depth=5, colsample=0.5, lam=1.0, min_hess=5.0,
                 semilla=0):
        self.__dict__.update({k: v for k, v in locals().items() if k != "self"})

    def fit(self, X, y, Xv=None, yv=None):
        PREP.ajustar(X)
        Xb = PREP.xb(X)
        yg = xp.asarray(y).astype(xp.float32)
        rng = np.random.default_rng(self.semilla)
        pos = np.nonzero(y == 1)[0]
        neg = np.nonzero(y == 0)[0]
        Fsum = xp.zeros(len(y), xp.float32)
        self.miembros = []
        for i in range(self.n_est):
            if i == 0:
                elegidos = rng.choice(neg, len(pos), replace=False)
            else:
                dur = a_np(_sig(Fsum[xp.asarray(neg)] / i))  # dureza de los negativos = p
                bordes = np.linspace(dur.min(), dur.max() + 1e-12, self.k_cubetas + 1)
                cub = np.clip(np.searchsorted(bordes, dur, side="right") - 1, 0, self.k_cubetas - 1)
                alfa = math.tan(i * math.pi / (2 * self.n_est))
                med = np.array([dur[cub == b].mean() if (cub == b).any() else np.inf for b in range(self.k_cubetas)])
                wb = np.where(np.isfinite(med), 1.0 / (med + alfa), 0.0)
                wb = wb / wb.sum()
                cuota = np.floor(wb * len(pos)).astype(int)
                elegidos = []
                for b in range(self.k_cubetas):
                    miembros_b = neg[cub == b]
                    if cuota[b] and len(miembros_b):
                        elegidos.append(rng.choice(miembros_b, min(cuota[b], len(miembros_b)), replace=False))
                elegidos = np.concatenate(elegidos)
            ii = xp.asarray(np.sort(np.concatenate([pos, elegidos])))
            m = BoostingNewton(n_rondas=self.n_r, tasa=self.tasa, max_depth=self.max_depth, colsample=self.colsample,
                               lam=self.lam, min_hess=self.min_hess, submuestra=0.8, semilla=self.semilla + i)
            m.fit(Xb[ii], yg[ii])
            Fsum += m.decision(Xb)
            self.miembros.append(m)
        self.mejor_it = None
        self.mejor_auc = _auc_np(yv, self.decision(Xv)) if Xv is not None else None
        return self

    def decision(self, X):
        Xb = PREP.xb(X)
        return a_np(sum(m.decision(Xb) for m in self.miembros) / len(self.miembros))


# ====================================================================== #24 DES
class DES:
    """Selección dinámica de ensambles: grupo de boostings propios entrenados en bootstraps del 80 % del
    entrenamiento; con el 20 % restante (DSEL) se estima la competencia local de cada miembro (log-verosimilitud
    media en los k vecinos más cercanos, en un espacio PCA). Predicción = suma ponderada softmax(τ·competencia)."""
    tipo_it = None

    def __init__(self, n_pool=8, n_r=300, tasa=0.05, max_depth=5, colsample=0.5, lam=1.0, min_hess=5.0, d_pca=10,
                 k_vec=50, tau=10.0, semilla=0):
        self.__dict__.update({k: v for k, v in locals().items() if k != "self"})

    def fit(self, X, y, Xv=None, yv=None):
        PREP.ajustar(X)
        Xb = PREP.xb(X)
        yg = xp.asarray(y).astype(xp.float32)
        rng = np.random.default_rng(self.semilla)
        n = len(y)
        es_dsel = np.zeros(n, bool)
        for c in (0, 1):
            ic = np.nonzero(y == c)[0]
            es_dsel[rng.choice(ic, int(0.2 * len(ic)), replace=False)] = True
        tr = np.nonzero(~es_dsel)[0]
        ds = xp.asarray(np.nonzero(es_dsel)[0])
        self.pool = []
        for b in range(self.n_pool):
            ii = xp.asarray(np.sort(rng.choice(tr, len(tr), replace=True)))
            m = BoostingNewton(n_rondas=self.n_r, tasa=self.tasa, max_depth=self.max_depth, colsample=self.colsample,
                               lam=self.lam, min_hess=self.min_hess, submuestra=0.8, semilla=self.semilla + b)
            m.fit(Xb[ii], yg[ii])
            self.pool.append(m)
        Z = PREP.z(X).astype(xp.float32)
        Zd = Z[ds]
        self.mu_pca = Zd.mean(0)
        _, _, Vt = xp.linalg.svd(Zd[:20000] - self.mu_pca, full_matrices=False)
        self.V = Vt[: self.d_pca].T.copy()
        self.Rd = (Zd - self.mu_pca) @ self.V
        Md = xp.stack([m.decision(Xb[ds]) for m in self.pool], 1)  # (nd, pool)
        yd = yg[ds][:, None]
        self.ll_d = (yd * _logsig(Md) + (1 - yd) * _logsig(-Md))  # log-verosimilitud por fila DSEL y miembro
        self.mejor_it = None
        self.mejor_auc = _auc_np(yv, self.decision(Xv)) if Xv is not None else None
        return self

    def decision(self, X):
        Xb = PREP.xb(X)
        R = (PREP.z(X).astype(xp.float32) - self.mu_pca) @ self.V
        M = xp.stack([m.decision(Xb) for m in self.pool], 1)
        out = xp.empty(len(X), xp.float32)
        nd2 = (self.Rd ** 2).sum(1)
        if gbdt.GPU:  # libera la caché del pool antes de la búsqueda de vecinos (no cambia el resultado)
            xp.get_default_memory_pool().free_all_blocks()
        bl = 256  # bloque de consultas (acota la memoria de la matriz de distancias; no cambia el resultado)
        for a in range(0, len(X), bl):
            Q = R[a:a + bl]
            D = (Q ** 2).sum(1)[:, None] - 2 * Q @ self.Rd.T + nd2[None, :]
            vec = xp.argpartition(D, self.k_vec, axis=1)[:, : self.k_vec]
            comp = self.ll_d[vec].mean(1)  # (q, pool)
            w = xp.exp(self.tau * (comp - comp.max(1, keepdims=True)))
            w = w / w.sum(1, keepdims=True)
            out[a:a + bl] = (w * M[a:a + bl]).sum(1)
        return a_np(out)


def _logsig(s):
    return -xp.logaddexp(0, -s)


# ====================================================================== #25 LMT
def _irls(Zf, y, beta0, prior, lam, iters=6, filas=None):
    """Regresión logística L2 por IRLS (Newton) encogida hacia `prior`. Zf incluye la columna de unos."""
    beta = beta0.copy()
    k = Zf.shape[1]
    reg = xp.full(k, lam, xp.float64)
    reg[0] = 1e-6
    for _ in range(iters):
        A = xp.diag(reg)
        b = -reg * (beta - prior)
        for a in range(0, Zf.shape[0], 40000):
            Zc = Zf[a:a + 40000].astype(xp.float32)
            p = _sig(Zc @ beta.astype(xp.float32))
            wv = xp.maximum(p * (1 - p), 1e-6)
            A = A + (Zc.T @ (Zc * wv[:, None])).astype(xp.float64)
            b = b + (Zc.T @ (y[a:a + 40000] - p)).astype(xp.float64)
        beta = beta + xp.linalg.solve(A, b)
    return beta


class LMT:
    """#25 Árbol de modelo logístico (Landwehr, Hall y Frank 2005): logística L2 en la raíz; un árbol poco profundo
    sobre el gradiente de esa logística; en cada hoja, una logística L2 encogida hacia la de la raíz."""
    tipo_it = None

    def __init__(self, prof=2, lam_raiz=10.0, lam_hoja=100.0, min_hoja=2000.0, semilla=0):
        self.__dict__.update({k: v for k, v in locals().items() if k != "self"})

    @staticmethod
    def _zf(X):
        Z = PREP.z(X)
        return xp.concatenate([xp.ones((Z.shape[0], 1), Z.dtype), Z], 1)

    def fit(self, X, y, Xv=None, yv=None):
        PREP.ajustar(X)
        Zf = self._zf(X)
        yg = xp.asarray(y).astype(xp.float32)
        k = Zf.shape[1]
        p0 = float(y.mean())
        b0 = xp.zeros(k, xp.float64)
        b0[0] = math.log(p0 / (1 - p0))
        self.b_raiz = _irls(Zf, yg, b0, xp.zeros(k, xp.float64), self.lam_raiz)
        F = (Zf.astype(xp.float32) @ self.b_raiz.astype(xp.float32))
        p = _sig(F)
        c = BoostingNewton(max_depth=self.prof, lam=1.0, min_hess=self.min_hoja * 0.15, colsample=1.0)
        Xb = PREP.xb(X)
        self.arbol, pos = c._crecer(Xb, p - yg, xp.maximum(p * (1 - p), 1e-6), xp.arange(len(y)), self.prof,
                                    np.random.default_rng(self.semilla), None)
        self.hojas = {}
        pos_np = a_np(pos)
        for hoja in np.unique(pos_np):
            ii = xp.asarray(np.nonzero(pos_np == hoja)[0])
            self.hojas[int(hoja)] = _irls(Zf[ii], yg[ii], self.b_raiz, self.b_raiz, self.lam_hoja)
        self.mejor_it = None
        self.mejor_auc = _auc_np(yv, self.decision(Xv)) if Xv is not None else None
        return self

    def decision(self, X):
        Zf = self._zf(X)
        pos = a_np(self.arbol.rutear(PREP.xb(X)))
        out = np.empty(len(X), np.float32)
        for hoja, beta in self.hojas.items():
            ii = np.nonzero(pos == hoja)[0]
            if len(ii):
                out[ii] = a_np(Zf[xp.asarray(ii)].astype(xp.float32) @ beta.astype(xp.float32))
        otros = ~np.isin(pos, list(self.hojas))
        if otros.any():
            out[otros] = a_np(Zf[xp.asarray(np.nonzero(otros)[0])].astype(xp.float32) @ self.b_raiz.astype(xp.float32))
        return out


# ====================================================================== #27 BART probit
class BART:
    """#27 BART probit (Chipman, George y McCulloch 2010): suma de m árboles con MCMC.
    Aumento de datos de Albert y Chib; movimientos crecer/podar con MH; hojas μ ~ N(0, σμ²), σμ = 3/(k√m);
    previa de corte α(1+d)^−β. Predicción = media posterior del latente (tras el quemado)."""
    tipo_it = None
    PROF = 6

    def __init__(self, m=50, n_iter=200, k=2.0, alfa=0.95, beta=2.0, min_hoja=5, semilla=0):
        self.__dict__.update({k_: v for k_, v in locals().items() if k_ != "self"})

    def _p_corte(self, d):
        return self.alfa * (1 + d) ** (-self.beta)

    def fit(self, X, y, Xv=None, yv=None):
        PREP.ajustar(X)
        Xb = PREP.xb(X)
        n, F = Xb.shape
        yg = xp.asarray(y).astype(xp.float32)
        rng = np.random.default_rng(self.semilla)
        rg = xp.random.default_rng(self.semilla) if gbdt.GPU else rng
        s2 = (3.0 / (self.k * math.sqrt(self.m))) ** 2
        self.off = float(ndtri(xp.asarray(max(min(float(y.mean()), 0.999), 0.001))))
        nodos = 2 ** (self.PROF + 1) - 1
        # estructura en CPU: feat, thr, hoja, existe; asignación de filas en GPU
        T = [dict(feat=np.zeros(nodos, np.int32), thr=np.zeros(nodos, np.int32), hoja=np.zeros(nodos, bool),
                  existe=np.zeros(nodos, bool), mu=np.zeros(nodos, np.float32)) for _ in range(self.m)]
        for t in T:
            t["hoja"][0] = t["existe"][0] = True
        asig = xp.zeros((self.m, n), xp.int32)
        f = xp.full(n, self.off, xp.float32)
        prof_de = np.floor(np.log2(np.arange(nodos) + 1)).astype(int)
        guardar = []
        quemado = self.n_iter // 2

        def loglik(nl, sl):
            return -0.5 * math.log(1 + nl * s2) + 0.5 * s2 * sl * sl / (1 + nl * s2)

        for it in range(self.n_iter):
            u = xp.clip(rg.random(n), 1e-7, 1 - 1e-7)
            c = ndtr(-f)
            z = xp.where(yg == 1, f + ndtri(xp.clip(c + u * (1 - c), 1e-7, 1 - 1e-7)),
                         f + ndtri(xp.clip(u * c, 1e-7, 1 - 1e-7))).astype(xp.float32)
            for j, t in enumerate(T):
                mu_g = xp.asarray(t["mu"])
                R = z - f + mu_g[asig[j]]
                hojas = np.nonzero(t["hoja"])[0]
                crecibles = hojas[prof_de[hojas] < self.PROF]
                nogs = [i for i in np.nonzero(t["existe"] & ~t["hoja"])[0]
                        if t["hoja"][2 * i + 1] and t["hoja"][2 * i + 2]]
                crecer = (len(nogs) == 0 or rng.random() < 0.5) and len(crecibles) > 0
                if crecer:
                    l = int(rng.choice(crecibles))
                    fj = int(rng.integers(F))
                    filas = asig[j] == l
                    col = Xb[:, fj]
                    validas = filas & (col != NAN_BIN)
                    if int(validas.sum()) < 2 * self.min_hoja:
                        pass
                    else:
                        lo = int(col[validas].min())
                        hi = int(col[validas].max())
                        if hi > lo:
                            tr = int(rng.integers(lo, hi))
                            izq = filas & ((col <= tr) | (col == NAN_BIN))
                            der = filas & ~izq
                            nL, nR = int(izq.sum()), int(der.sum())
                            if nL >= self.min_hoja and nR >= self.min_hoja:
                                sL, sR = float(R[izq].sum()), float(R[der].sum())
                                d = prof_de[l]
                                pc = self._p_corte(d)
                                pc2 = self._p_corte(d + 1)
                                lr = loglik(nL, sL) + loglik(nR, sR) - loglik(nL + nR, sL + sR)
                                lr += math.log(pc) + 2 * math.log(1 - pc2) - math.log(1 - pc)
                                n_nog_desp = len([i for i in nogs if i != (l - 1) // 2 or l == 0]) + 1
                                lr += math.log(len(crecibles) * F * (hi - lo)) - math.log(n_nog_desp)
                                if math.log(rng.random() + 1e-300) < lr:
                                    t["feat"][l], t["thr"][l], t["hoja"][l] = fj, tr, False
                                    for ch in (2 * l + 1, 2 * l + 2):
                                        t["hoja"][ch] = t["existe"][ch] = True
                                    asig[j] = xp.where(izq, 2 * l + 1, xp.where(der, 2 * l + 2, asig[j]))
                elif nogs:
                    i = int(rng.choice(nogs))
                    a_, b_ = 2 * i + 1, 2 * i + 2
                    mA, mB = asig[j] == a_, asig[j] == b_
                    nL, nR = int(mA.sum()), int(mB.sum())
                    sL, sR = float(R[mA].sum()), float(R[mB].sum())
                    d = prof_de[i]
                    pc, pc2 = self._p_corte(d), self._p_corte(d + 1)
                    lr = loglik(nL + nR, sL + sR) - loglik(nL, sL) - loglik(nR, sR)
                    lr -= math.log(pc) + 2 * math.log(1 - pc2) - math.log(1 - pc)
                    col = Xb[:, int(t["feat"][i])]
                    val_i = (mA | mB) & (col != NAN_BIN)
                    n_thr = max(int(col[val_i].max()) - int(col[val_i].min()), 1) if bool(val_i.any()) else 1
                    n_crec_desp = len(crecibles) - sum(prof_de[c_] < self.PROF for c_ in (a_, b_)) + 1
                    lr += math.log(len(nogs)) - math.log(max(n_crec_desp, 1) * F * n_thr)
                    if math.log(rng.random() + 1e-300) < lr:
                        t["hoja"][i] = True
                        t["hoja"][a_] = t["hoja"][b_] = t["existe"][a_] = t["existe"][b_] = False
                        asig[j] = xp.where(mA | mB, i, asig[j])
                # Gibbs de las hojas
                S = suma_dispersa(nodos, asig[j], R)
                N = suma_dispersa(nodos, asig[j], xp.ones(n, xp.float32))
                var = 1.0 / (N + 1.0 / s2)
                mu = (var * S + xp.sqrt(var) * rg.standard_normal(nodos).astype(xp.float32)).astype(xp.float32)
                t["mu"] = a_np(mu) * t["hoja"]
                f = (z - R) + xp.asarray(t["mu"])[asig[j]]  # resto de árboles + árbol j nuevo
            if it >= quemado:
                guardar.append([(t["feat"].copy(), t["thr"].copy(), t["hoja"].copy(), t["mu"].copy()) for t in T])
        self.muestras = guardar
        self.mejor_it = None
        self.mejor_auc = _auc_np(yv, self.decision(Xv)) if Xv is not None else None
        return self

    def decision(self, X):
        Xb = PREP.xb(X)
        n = Xb.shape[0]
        filas = xp.arange(n)
        tot = xp.zeros(n, xp.float32)
        for muestra in self.muestras:
            for feat, thr, hoja, mu in muestra:
                pos = xp.zeros(n, xp.int32)
                fg, tg, hg = xp.asarray(feat), xp.asarray(thr), xp.asarray(hoja)
                for _ in range(self.PROF):
                    b = Xb[filas, fg[pos]].astype(xp.int32)
                    izq = (b <= tg[pos]) | (b == NAN_BIN)
                    pos = xp.where(hg[pos], pos, 2 * pos + 1 + (~izq).astype(xp.int32))
                tot += xp.asarray(mu)[pos]
        return a_np(self.off + tot / len(self.muestras))


# ====================================================================== #28 P-splines por componentes
def _base_bspline(K, grado=3, n=NAN_BIN):
    """Base B-spline (n × K) sobre los índices de cubeta 0..n-1 (nudos equiespaciados)."""
    x = np.arange(n, dtype=np.float64)
    interior = K - grado
    nudos = np.linspace(0, n - 1, interior + 1)
    h = nudos[1] - nudos[0]
    t = np.concatenate([nudos[0] - h * np.arange(grado, 0, -1), nudos, nudos[-1] + h * np.arange(1, grado + 1)])
    B = np.zeros((n, len(t) - 1))
    for i in range(len(t) - 1):
        B[:, i] = (x >= t[i]) & (x < t[i + 1])
    B[-1, interior + grado - 1] = 1.0
    for d in range(1, grado + 1):
        Bn = np.zeros((n, len(t) - 1 - d))
        for i in range(len(t) - 1 - d):
            a = (x - t[i]) / (t[i + d] - t[i]) * B[:, i]
            b = (t[i + d + 1] - x) / (t[i + d + 1] - t[i + 1]) * B[:, i + 1]
            Bn[:, i] = a + b
        B = Bn
    return B[:, :K]


class PSplines:
    """#28 Boosting por componentes con P-splines (Bühlmann y Hothorn 2007; Eilers y Marx): en cada ronda se ajusta,
    para cada variable, una P-spline cúbica (penalización de diferencias de orden 2) al paso de Newton sobre sus
    cubetas, y se suma solo la de mayor reducción de la pérdida. Los faltantes tienen su propio coeficiente."""
    tipo_it = "rondas"

    def __init__(self, n_rondas=5000, tasa=0.1, K=12, lam=10.0, submuestra=0.8, paciencia=100, eval_cada=10,
                 semilla=0, **_):
        self.__dict__.update({k: v for k, v in locals().items() if k not in ("self", "_")})

    def fit(self, X, y, Xv=None, yv=None):
        PREP.ajustar(X)
        Xb = PREP.xb(X)
        n, F = Xb.shape
        yg = xp.asarray(y).astype(xp.float32)
        B = xp.asarray(_base_bspline(self.K), dtype=xp.float32)  # (255, K)
        D = np.diff(np.eye(self.K), 2, axis=0)
        Pen = xp.asarray(self.lam * D.T @ D, dtype=xp.float32)
        p0 = float(y.mean())
        self.base = math.log(p0 / (1 - p0))
        self.tabla = xp.zeros((F, 256), xp.float32)
        Fx = xp.full(n, self.base, xp.float32)
        hay_val = Xv is not None
        if hay_val:
            Xbv = PREP.xb(Xv)
            Fv = xp.full(len(Xv), self.base, xp.float32)
        motor = BoostingNewton()
        rng = np.random.default_rng(self.semilla)
        mejor, mejor_it, sin = -1, 0, 0
        mejor_tabla = None
        for it in range(self.n_rondas):
            p = _sig(Fx)
            g, h = p - yg, xp.maximum(p * (1 - p), 1e-6)
            idx = xp.asarray(np.nonzero(rng.random(n) < self.submuestra)[0])
            H = motor._hist(xp.zeros(len(idx), xp.int32), 1, Xb[idx], g[idx], h[idx])[0]  # (F, 256, 2)
            Gb, Hb = H[:, :NAN_BIN, 0], H[:, :NAN_BIN, 1]
            A = (B.T[None] * Hb[:, None, :]) @ B + Pen[None] + 1e-4 * xp.eye(self.K, dtype=xp.float32)[None]
            bv = -(Gb @ B)  # (F, K)
            th = xp.linalg.solve(A, bv[..., None])[..., 0]
            gan = 0.5 * (bv * th).sum(1)
            Gn, Hn = H[:, NAN_BIN, 0], H[:, NAN_BIN, 1]
            dn = -Gn / (Hn + 1.0)
            gan = gan + 0.5 * Gn * Gn / (Hn + 1.0)
            j = int(xp.argmax(gan))
            fila = xp.zeros(256, xp.float32)
            fila[:NAN_BIN] = B @ th[j]
            fila[NAN_BIN] = dn[j]
            fila *= self.tasa
            self.tabla[j] += fila
            Fx += fila[Xb[:, j].astype(xp.int32)]
            if hay_val:
                Fv += fila[Xbv[:, j].astype(xp.int32)]
                if (it + 1) % self.eval_cada == 0:
                    a = _auc_np(yv, Fv)
                    if a > mejor:
                        mejor, mejor_it, sin = a, it + 1, 0
                        mejor_tabla = self.tabla.copy()
                    else:
                        sin += self.eval_cada
                        if sin >= self.paciencia:
                            break
        if hay_val and mejor_tabla is not None:
            self.tabla = mejor_tabla
        self.mejor_auc, self.mejor_it = (mejor, mejor_it) if hay_val else (None, self.n_rondas)
        return self

    def decision(self, X):
        Xb = PREP.xb(X).astype(xp.int32)
        usadas = xp.nonzero(xp.abs(self.tabla).sum(1) > 0)[0]
        F = xp.full(Xb.shape[0], self.base, xp.float32)
        for j in a_np(usadas):
            F += self.tabla[j][Xb[:, j]]
        return a_np(F)


# ====================================================================== espacios de búsqueda (TPE)
def base(t):
    """Mismo espacio que XGBoost (ajuste_xgb.py / ajuste_propio.py)."""
    return dict(
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


def _boost(extra):
    def esp(t):
        p = base(t)
        p.update(extra(t) if callable(extra) else extra)
        return p
    return esp


def _pequeno(t):
    return dict(n_r=t.suggest_int("n_r", 100, 600, log=True), tasa=t.suggest_float("tasa", 0.02, 0.2, log=True),
                max_depth=t.suggest_int("max_depth", 3, 7), colsample=t.suggest_float("colsample", 0.2, 1.0),
                lam=t.suggest_float("lam", 1e-2, 100, log=True), min_hess=t.suggest_float("min_hess", 1, 100, log=True))


def _bosque(t):
    return dict(n_arboles=t.suggest_int("n_arboles", 30, 200, log=True), prof=t.suggest_int("prof", 6, 10),
                col_arbol=t.suggest_float("col_arbol", 0.1, 0.5), col_nodo=t.suggest_float("col_nodo", 0.1, 1.0),
                min_hoja=t.suggest_float("min_hoja", 1, 200, log=True))


def _ga2m(t):
    p = base(t)
    p.update(una_variable=True, max_depth=t.suggest_int("prof_ebm", 1, 3), colsample=1.0,
             pares_ga2m=t.suggest_int("pares_ga2m", 5, 30), rondas_main=t.suggest_int("rondas_main", 300, 3000, log=True))
    return p


CATALOGO = {
    # boostings (mismo espacio base que XGBoost + sus parámetros)
    "ktboost": (lambda **p: Booster(**p), _boost(lambda t: dict(
        nucleo_D=t.suggest_categorical("nucleo_D", [64, 128, 256]), nucleo_ls=t.suggest_float("nucleo_ls", 5, 100, log=True),
        nucleo_lam=t.suggest_float("nucleo_lam", 1, 1000, log=True))), "rondas"),
    "rotaciones": (lambda **p: Booster(vistas="rotacion", **p), _boost(lambda t: dict(
        n_vistas=t.suggest_int("n_vistas", 2, 4), vista_m=t.suggest_int("vista_m", 32, 200, log=True),
        bloque=t.suggest_int("bloque", 2, 8))), "rondas"),
    "oblicuos": (lambda **p: Booster(vistas="oblicuo", **p), _boost(lambda t: dict(
        n_vistas=t.suggest_int("n_vistas", 2, 4), vista_m=t.suggest_int("vista_m", 32, 200, log=True),
        densidad=t.suggest_int("densidad", 2, 4))), "rondas"),
    "rgf": (lambda **p: Booster(**p), _boost(lambda t: dict(
        fc_cada=t.suggest_int("fc_cada", 20, 200, log=True), tasa_fc=t.suggest_float("tasa_fc", 0.1, 1.0))), "rondas"),
    "infiniteboost": (lambda **p: Booster(**p), _boost(lambda t: dict(
        paso="infinito", tasa_c=t.suggest_float("tasa_c", 0.1, 1.0))), "rondas"),
    "rankboost": (lambda **p: Booster(**p), _boost(lambda t: dict(
        perdida="pares", pares_k=t.suggest_int("pares_k", 1, 10))), "rondas"),
    "trboost": (lambda **p: Booster(**p), _boost(lambda t: dict(
        paso="confianza", mu0=t.suggest_float("mu0", 1e-2, 100, log=True))), "rondas"),
    "nesterov": (lambda **p: Booster(**p), _boost(dict(paso="nesterov")), "rondas"),
    "focal": (lambda **p: Booster(**p), _boost(lambda t: dict(
        perdida="focal", gamma_focal=t.suggest_float("gamma_focal", 0.5, 5.0))), "rondas"),
    "ngboost": (lambda **p: Booster(**p), _boost(dict(perdida="natural")), "rondas"),
    "ga2m": (lambda **p: Booster(**p), _ga2m, "rondas"),
    "psplines": (PSplines, lambda t: dict(tasa=t.suggest_float("tasa", 0.05, 0.5, log=True), K=t.suggest_int("K", 6, 30),
                                          lam=t.suggest_float("lam", 0.1, 1000, log=True),
                                          submuestra=t.suggest_float("submuestra", 0.5, 1.0)), "rondas"),
    # bosques y otros (espacio propio)
    "extra_trees": (lambda **p: Bosque(aleatorio=True, **p), _bosque, None),
    "bosque_rotacion": (lambda **p: Bosque(rotacion=True, **p), lambda t: dict(
        n_arboles=t.suggest_int("n_arboles", 20, 100, log=True), prof=t.suggest_int("prof", 6, 10),
        col_arbol=t.suggest_float("col_arbol", 0.2, 0.6), col_nodo=t.suggest_float("col_nodo", 0.2, 1.0),
        min_hoja=t.suggest_float("min_hoja", 1, 200, log=True), rot_k=t.suggest_int("rot_k", 3, 10)), None),
    "bosque_profundo": (BosqueProfundo, lambda t: dict(
        n_arboles=t.suggest_int("n_arboles", 20, 80, log=True), prof=t.suggest_int("prof", 6, 10),
        col_arbol=t.suggest_float("col_arbol", 0.1, 0.5), col_nodo=t.suggest_float("col_nodo", 0.1, 1.0),
        min_hoja=t.suggest_float("min_hoja", 1, 200, log=True), niveles_max=t.suggest_int("niveles_max", 1, 4)),
        "niveles"),
    "lce": (LCE, lambda t: dict(n_bolsas=t.suggest_int("n_bolsas", 2, 6), prof_cascada=t.suggest_int("prof_cascada", 1, 2),
                                min_filas=t.suggest_int("min_filas", 1000, 10000, log=True), **_pequeno(t)), None),
    "spe": (SPE, lambda t: dict(n_est=t.suggest_int("n_est", 5, 20), k_cubetas=t.suggest_int("k_cubetas", 5, 20),
                                **_pequeno(t)), None),
    "des": (DES, lambda t: dict(n_pool=t.suggest_int("n_pool", 5, 15), d_pca=t.suggest_int("d_pca", 5, 30),
                                k_vec=t.suggest_int("k_vec", 20, 200, log=True), tau=t.suggest_float("tau", 1, 50, log=True),
                                **_pequeno(t)), None),
    "lmt": (LMT, lambda t: dict(prof=t.suggest_int("prof", 1, 3), lam_raiz=t.suggest_float("lam_raiz", 0.1, 1000, log=True),
                                lam_hoja=t.suggest_float("lam_hoja", 0.1, 10000, log=True),
                                min_hoja=t.suggest_float("min_hoja", 500, 5000, log=True)), None),
    "bart": (BART, lambda t: dict(m=t.suggest_int("m", 20, 200, log=True), n_iter=t.suggest_int("n_iter", 100, 300, log=True),
                                  k=t.suggest_float("k", 1.0, 3.0), alfa=t.suggest_float("alfa", 0.8, 0.99),
                                  beta=t.suggest_float("beta", 1.0, 3.0)), None),
}

# #15 bstacking no está aquí: es un metamodelo sobre predicciones OOF (superlearner.py --logistica).
ORDEN = ["ktboost", "rotaciones", "oblicuos", "rgf", "infiniteboost", "lce", "rankboost", "trboost", "nesterov",
         "focal", "bosque_profundo", "bosque_rotacion", "extra_trees", "spe", "des", "lmt", "bart", "psplines",
         "ga2m", "ngboost"]


def construir(nombre, params, n_rondas=None):
    """Instancia el modelo; para los iterativos, n_rondas fija las rondas sin parada temprana."""
    cons, _, tipo = CATALOGO[nombre]
    p = dict(params)
    if tipo == "rondas":
        p["n_rondas"] = n_rondas or 5000
    elif tipo == "niveles" and n_rondas:
        p["n_rondas"] = n_rondas
    return cons(**p)
