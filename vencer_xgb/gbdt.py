"""Motor propio de boosting de árboles (Newton) con la API de NumPy.

`xp` es numpy o cupy (sustituto de NumPy en GPU); el algoritmo es el mismo.
Variantes (cada una corresponde a un ítem del Top 30):
  #4  Newton boosting (base, crecimiento por niveles, histogramas, dirección por defecto para NaN)
  #2  hojas lineales (ridge local de Newton con las variables del camino)        hojas_lineales=True
  #3  profundidad heterogénea aleatoria por ronda                                 prof_min < max_depth
  #7  DART (abandono de árboles)                                                   dart_tasa > 0
  #6  árboles olvidadizos (simétricos)                                             olvidadizo=True
  #12 muestreo de varianza mínima (MVS)                                            muestreo="mvs"
  #26 GOSS                                                                         muestreo="goss"
  #29 GA2M/EBM: un árbol por variable, ciclo de variables, luego pares             una_variable=True, pares_k>0
Ampliación (21 restantes del Top 30):
  #5  KTBoost: en cada ronda, árbol o ridge sobre rasgos de Fourier (núcleo RBF)   fit(..., Phi=...)
  #8/#9 vistas: rotaciones por bloques / proyecciones oblicuas dispersas           fit(Xb=[vista0, vista1, ...])
  #10 RGF: paso totalmente correctivo sobre las hojas de todos los árboles         fc_cada > 0
  #13 InfiniteBoost: F = base + c · promedio de árboles                            paso="infinito"
  #16 RankBoost/AUC: pérdida logística por pares                                   perdida="pares"
  #17 TRBoost: región de confianza                                                 paso="confianza"
  #18 Nesterov (Biau, Cadre y Rouvière 2019)                                       paso="nesterov"
  #19 pérdida focal                                                                perdida="focal"
  #30 NGBoost: gradiente natural + búsqueda lineal                                 perdida="natural"
  #22 cortes aleatorios y mtry por nodo (para los bosques)                         corte_aleatorio, col_nodo
"""
import math
import numpy as np

try:
    import cupy as cp
    import cupyx
    xp = cp
    GPU = True
    # Tope del pool (no cambia ningún cálculo): al llegarlo, CuPy libera bloques en caché antes de pedir más.
    # Sin él, en una GPU de 4 GB el controlador de Windows desborda a memoria compartida y todo va ~50x más lento.
    try:
        _libre, _total = cp.cuda.runtime.memGetInfo()
        cp.get_default_memory_pool().set_limit(size=int(_total * 0.80))
    except Exception:  # pragma: no cover
        pass
except Exception:  # pragma: no cover
    cp = None
    xp = np
    GPU = False

NAN_BIN = 255


def a_np(a):
    return cp.asnumpy(a) if GPU and isinstance(a, cp.ndarray) else np.asarray(a)


def suma_dispersa(n, idx, w):
    """bincount con pesos en float32 (atómicos en GPU)."""
    if GPU:
        out = xp.zeros(n, dtype=xp.float32)
        cupyx.scatter_add(out, idx, w)
        return out
    return np.bincount(idx, weights=w, minlength=n).astype(np.float32)


def auc(y, s):
    y = xp.asarray(y)
    s = xp.asarray(s, dtype=xp.float64)
    orden = xp.argsort(s)
    rango = xp.empty_like(s)
    rango[orden] = xp.arange(1, len(s) + 1, dtype=xp.float64)
    # empates: promedio de rangos
    ss = s[orden]
    _, inv, cnt = xp.unique(ss, return_inverse=True, return_counts=True)
    if len(cnt) < len(s):
        csum = xp.cumsum(cnt)
        prom = csum - (cnt - 1) / 2.0
        rango[orden] = prom[inv]
    npos = float(y.sum())
    nneg = len(y) - npos
    return float((rango[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


class Binarizador:
    def __init__(self, max_bins=255, semilla=0):
        self.max_bins = max_bins
        self.semilla = semilla

    def fit(self, X):
        rng = np.random.default_rng(self.semilla)
        idx = rng.choice(len(X), size=min(len(X), 200_000), replace=False)
        S = X[idx]
        self.bordes = []
        for j in range(X.shape[1]):
            v = S[:, j]
            v = v[~np.isnan(v)]
            if len(v) == 0:
                self.bordes.append(np.array([], dtype=np.float32))
                continue
            u = np.unique(v)
            if len(u) <= self.max_bins - 1:
                b = (u[:-1] + u[1:]) / 2
            else:
                q = np.quantile(v, np.linspace(0, 1, self.max_bins)[1:-1])
                b = np.unique(q)
            self.bordes.append(b.astype(np.float32))
        return self

    def transform(self, X):
        Xb = np.empty(X.shape, dtype=np.uint8)
        for j, b in enumerate(self.bordes):
            v = X[:, j]
            col = np.searchsorted(b, v, side="left").astype(np.uint8)
            col[np.isnan(v)] = NAN_BIN
            Xb[:, j] = col
        return Xb


def _umbral_suave(G, alpha):
    if alpha <= 0:
        return G
    return xp.sign(G) * xp.maximum(xp.abs(G) - alpha, 0)


class Arbol:
    """Árbol en disposición de montículo (nodo i → hijos 2i+1, 2i+2)."""

    def __init__(self, prof):
        m = 2 ** (prof + 1) - 1
        self.prof = prof
        self.feat = xp.zeros(m, dtype=xp.int32)
        self.thr = xp.zeros(m, dtype=xp.int32)
        self.izq_nan = xp.zeros(m, dtype=xp.bool_)
        self.hoja = xp.ones(m, dtype=xp.bool_)
        self.valor = xp.zeros(m, dtype=xp.float32)
        self.lin_feat = None  # (m, k) int32
        self.lin_coef = None  # (m, k+1) float32
        self.peso = 1.0
        self.vista = 0
        self.ganancia = 0.0

    def rutear(self, Xb):
        n = Xb.shape[0]
        pos = xp.zeros(n, dtype=xp.int32)
        filas = xp.arange(n)
        for _ in range(self.prof):
            f = self.feat[pos]
            b = Xb[filas, f].astype(xp.int32)
            izq = (b <= self.thr[pos]) & (b != NAN_BIN) | ((b == NAN_BIN) & self.izq_nan[pos])
            nuevo = 2 * pos + 1 + (~izq).astype(xp.int32)
            pos = xp.where(self.hoja[pos], pos, nuevo)
        return pos

    def valor_filas(self, pos, Z=None):
        v = self.valor[pos]
        if self.lin_feat is not None:
            k = self.lin_feat.shape[1]
            filas = xp.arange(len(pos))
            zf = Z[filas[:, None], self.lin_feat[pos]].astype(xp.float32)  # (n,k)
            c = self.lin_coef[pos]
            v = c[:, 0] + (c[:, 1:] * zf).sum(1)
        return v * self.peso


class Nucleo:
    """Aprendiz de núcleo de KTBoost: ridge de Newton sobre rasgos aleatorios de Fourier."""
    lin_feat = None

    def __init__(self, coef):
        self.coef = coef
        self.peso = 1.0


def _vista(Xb, k):
    return Xb[k] if isinstance(Xb, (list, tuple)) else Xb


def _logsig(s):
    return -xp.logaddexp(0, -s)


class BoostingNewton:
    def __init__(self, n_rondas=1000, tasa=0.05, max_depth=6, prof_min=None, lam=1.0, alpha=0.0,
                 gamma=0.0, min_hess=1.0, submuestra=0.8, colsample=0.5, peso_pos=1.0,
                 hojas_lineales=False, lam_lin=10.0, dart_tasa=0.0, olvidadizo=False,
                 muestreo="bernoulli", una_variable=False, semilla=0, paciencia=100,
                 eval_cada=10, verbose=False, max_delta=0.0,
                 perdida="logistica", gamma_focal=2.0, pares_k=4, paso="newton", mu0=1.0, tasa_c=0.5,
                 fc_cada=0, tasa_fc=0.5, nucleo_lam=10.0, col_nodo=1.0, corte_aleatorio=False,
                 pares_ga2m=0, rondas_main=0):
        self.__dict__.update({k: v for k, v in locals().items() if k != "self"})
        self._lam_extra = 0.0
        self._rng_gpu = None

    # ------------------------------------------------------------------ utilidades
    def _muestra(self, rng, g, h, n):
        if self.submuestra >= 1.0:
            return xp.arange(n), None
        if self.muestreo == "bernoulli":
            return xp.nonzero(rng.random(n) < self.submuestra)[0], None
        if self.muestreo == "goss":
            a = self.submuestra * 0.5
            b = self.submuestra - a
            ag = xp.abs(g)
            k = int(a * n)
            umbral = xp.partition(ag, n - k)[n - k]
            top = ag >= umbral
            resto = (~top) & (rng.random(n) < b / (1 - a))
            idx = xp.nonzero(top | resto)[0]
            mult = xp.where(top[idx], 1.0, (1 - a) / b).astype(xp.float32)
            return idx, mult
        if self.muestreo == "mvs":
            # probabilidad proporcional a sqrt(g^2 + lambda h^2), recortada en 1
            r = xp.sqrt(g * g + 0.1 * h * h)
            lo, hi = 0.0, float(r.max())
            for _ in range(30):
                mu = (lo + hi) / 2
                if float(xp.minimum(r / mu, 1).sum()) > self.submuestra * n:
                    lo = mu
                else:
                    hi = mu
            p = xp.minimum(r / hi, 1.0)
            sel = rng.random(n) < p
            idx = xp.nonzero(sel)[0]
            return idx, (1.0 / p[idx]).astype(xp.float32)
        raise ValueError(self.muestreo)

    def _hist(self, slot, n_slots, Xb_f, g, h):
        """Histograma (n_slots, F', 256, 2) en trozos de variables."""
        n, Fp = Xb_f.shape
        out = xp.zeros((n_slots, Fp, 256, 2), dtype=xp.float32)
        trozo = max(1, int(30_000_000 // max(n, 1)))
        for a in range(0, Fp, trozo):
            b = min(Fp, a + trozo)
            fc = b - a
            clave = ((slot[:, None] * fc + xp.arange(fc, dtype=xp.int32)[None, :]) * 256
                     + Xb_f[:, a:b].astype(xp.int32)).ravel()
            tam = n_slots * fc * 256
            hg = suma_dispersa(tam, clave, xp.broadcast_to(g[:, None], (n, fc)).ravel())
            hh = suma_dispersa(tam, clave, xp.broadcast_to(h[:, None], (n, fc)).ravel())
            out[:, a:b, :, 0] = hg.reshape(n_slots, fc, 256)
            out[:, a:b, :, 1] = hh.reshape(n_slots, fc, 256)
        return out

    def _mejores_cortes(self, H, mascara_f):
        """Igual que _cortes_trozo, por trozos de nodos para acotar la memoria (salvo olvidadizo: es global)."""
        Lnod, Fp = H.shape[0], H.shape[1]
        paso = max(1, int(6_000_000 // max(Fp * 256, 1)))
        if self.olvidadizo or Lnod <= paso:
            return self._cortes_trozo(H, mascara_f)
        partes = [self._cortes_trozo(H[a:a + paso], mascara_f) for a in range(0, Lnod, paso)]
        return tuple(xp.concatenate([p[i] for p in partes]) for i in range(4))

    def _cortes_trozo(self, H, mascara_f):
        """H: (L, F', 256, 2). Devuelve ganancia, f, t, izq_nan por nodo (o global si olvidadizo)."""
        lam, alpha, mh = self.lam, self.alpha, self.min_hess
        tot = H[:, :1].sum(axis=2)  # (L,1,2)
        Gt, Ht = tot[..., 0], tot[..., 1]  # (L,1)
        val = H[:, :, :NAN_BIN, :]
        cum = xp.cumsum(val, axis=2)[:, :, :-1, :]  # cortes t=0..253
        nanv = H[:, :, NAN_BIN, :][:, :, None, :]
        sc = lambda G, Hh: _umbral_suave(G, alpha) ** 2 / (Hh + lam)
        base = sc(Gt, Ht)[:, :, None]
        gan = []
        for izq_nan in (False, True):
            L = cum + nanv if izq_nan else cum
            GL, HL = L[..., 0], L[..., 1]
            GR, HR = Gt[:, :, None] - GL, Ht[:, :, None] - HL
            g = sc(GL, HL) + sc(GR, HR) - base
            g = xp.where((HL >= mh) & (HR >= mh), g, -xp.inf)
            gan.append(g)
        gan = xp.stack(gan, axis=-1)  # (L,F',254,2)
        gan = xp.where(mascara_f[None, :, None, None], gan, -xp.inf)
        Lnod = gan.shape[0]
        if self.corte_aleatorio:  # #22: un umbral al azar por nodo y variable
            occ = H[:, :, :NAN_BIN, 1] > 0
            prim = xp.argmax(occ, axis=2)
            ult = (NAN_BIN - 1) - xp.argmax(occ[:, :, ::-1], axis=2)
            u = self._rng_gpu.random(prim.shape)
            t_al = xp.clip(prim + xp.floor(u * xp.maximum(ult - prim, 0)).astype(prim.dtype), 0, NAN_BIN - 2)
            gan = xp.where(xp.arange(NAN_BIN - 1)[None, None, :, None] == t_al[:, :, None, None], gan, -xp.inf)
        if self.col_nodo < 1.0:  # mtry por nodo
            m_nodo = self._rng_gpu.random((Lnod, gan.shape[1])) < self.col_nodo
            gan = xp.where(m_nodo[:, :, None, None], gan, -xp.inf)
        if self.olvidadizo:
            gsum = xp.where(xp.isfinite(gan), gan, 0).sum(0)
            valido = xp.isfinite(gan).all(0)
            gsum = xp.where(valido, gsum, -xp.inf)
            k = int(xp.argmax(gsum.ravel()))
            f, t, d = np.unravel_index(k, gsum.shape)
            gn = gan[:, f, t, d]
            return gn, xp.full(Lnod, f, xp.int32), xp.full(Lnod, t, xp.int32), xp.full(Lnod, bool(d))
        plano = gan.reshape(Lnod, -1)
        k = xp.argmax(plano, axis=1)
        gn = plano[xp.arange(Lnod), k]
        f, t, d = xp.unravel_index(k, gan.shape[1:])
        return gn, f.astype(xp.int32), t.astype(xp.int32), d.astype(xp.bool_)

    # ------------------------------------------------------------------ un árbol
    def _crecer(self, Xb, g, h, idx, prof, rng, feats_permitidas):
        F = Xb.shape[1]
        arbol = Arbol(prof)
        if feats_permitidas is None:
            m = max(1, int(round(self.colsample * F)))
            fsel = xp.asarray(np.sort(rng.choice(F, m, replace=False)).astype(np.int32))
        else:
            fsel = xp.asarray(feats_permitidas)
        mascara = xp.ones(len(fsel), dtype=xp.bool_)
        Xs = Xb[idx][:, fsel]
        gi, hi = g[idx], h[idx]
        pos = xp.zeros(len(idx), dtype=xp.int32)  # heap id
        activo = xp.ones(len(idx), dtype=xp.bool_)
        H_padre = None
        for nivel in range(prof):
            base = 2 ** nivel - 1
            L = 2 ** nivel
            loc = xp.clip(pos - base, 0, L - 1)
            if nivel == 0:
                Hn = self._hist(xp.zeros(len(idx), xp.int32), 1, Xs, gi, hi)
            else:
                # truco de la resta: se calcula el hijo con menos filas de cada par
                cnt = xp.bincount(loc[activo], minlength=L)
                par = cnt.reshape(-1, 2)
                calc_der = par[:, 1] < par[:, 0]  # (L/2,)
                hijo = loc % 2
                p_id = loc // 2
                usar = activo & (hijo == calc_der[p_id].astype(xp.int32))
                Hc = self._hist(p_id[usar].astype(xp.int32), L // 2, Xs[usar], gi[usar], hi[usar])
                otro = H_padre - Hc
                Hn = xp.empty((L,) + Hc.shape[1:], dtype=xp.float32)
                cd = calc_der[:, None, None, None]
                Hn[0::2] = xp.where(cd, otro, Hc)
                Hn[1::2] = xp.where(cd, Hc, otro)
                # nodos que ya eran hoja: histograma cero
                vivo = arbol_vivo
                Hn = Hn * vivo[:, None, None, None]
            gn, f, t, d = self._mejores_cortes(Hn, mascara)
            corta = (gn > self.gamma) & xp.isfinite(gn)
            if nivel > 0:
                corta = corta & arbol_vivo
            ids = base + xp.arange(L)
            arbol.hoja[ids] = ~corta
            arbol.feat[ids] = fsel[f]
            arbol.thr[ids] = t
            arbol.izq_nan[ids] = d
            if not bool(corta.any()):
                break
            # rutear filas
            c_fila = corta[loc] & activo
            fl = f[loc]
            b = Xs[xp.arange(len(idx)), fl].astype(xp.int32)
            izq = ((b <= t[loc]) & (b != NAN_BIN)) | ((b == NAN_BIN) & d[loc])
            pos = xp.where(c_fila, 2 * pos + 1 + (~izq).astype(xp.int32), pos)
            activo = c_fila
            arbol_vivo = xp.repeat(corta, 2)
            H_padre = Hn
        # hojas: los nodos restantes por debajo de un corte del último nivel son hojas
        m = len(arbol.hoja)
        Gh = suma_dispersa(m, pos, gi)
        Hh = suma_dispersa(m, pos, hi)
        Gs = _umbral_suave(Gh, self.alpha)
        v = -Gs / (Hh + self.lam + self._lam_extra)
        if self.max_delta > 0:
            v = xp.clip(v, -self.max_delta, self.max_delta)
        arbol.valor = (v * self.tasa).astype(xp.float32)
        arbol.ganancia = float(0.5 * (Gs * Gs / (Hh + self.lam)).sum())
        arbol._G, arbol._H = Gh, Hh
        return arbol, pos

    def _hojas_lineales(self, arbol, pos, idx, Z, g, h):
        """Hoja = ridge de Newton sobre las variables del camino (Shi, Li y Li 2019)."""
        m = len(arbol.hoja)
        k = arbol.prof
        # variables del camino de cada nodo del montículo
        lf = np.zeros((m, k), dtype=np.int32)
        dep = np.zeros(m, dtype=np.int32)
        feat = a_np(arbol.feat)
        for i in range(1, m):
            p = (i - 1) // 2
            lf[i] = lf[p]
            prof_i = int(math.floor(math.log2(i + 1)))
            lf[i, prof_i - 1] = feat[p]
            dep[i] = prof_i
        lf = xp.asarray(lf)
        dep = xp.asarray(dep)
        msk = (xp.arange(k)[None, :] < dep[pos][:, None]).astype(xp.float32)
        zf = Z[idx[:, None], lf[pos]].astype(xp.float32) * msk  # (n,k)
        zf = xp.concatenate([xp.ones((len(idx), 1), xp.float32), zf], axis=1)  # (n,k+1)
        gi, hi = g[idx], h[idx]
        kk = k + 1
        A = suma_dispersa(m * kk * kk, (pos[:, None] * kk * kk + xp.arange(kk * kk)[None, :]).ravel(),
                          (hi[:, None, None] * zf[:, :, None] * zf[:, None, :]).reshape(len(idx), -1).ravel())
        A = A.reshape(m, kk, kk)
        bvec = suma_dispersa(m * kk, (pos[:, None] * kk + xp.arange(kk)[None, :]).ravel(),
                             (-gi[:, None] * zf).ravel()).reshape(m, kk)
        reg = xp.full(kk, self.lam_lin, dtype=xp.float32)
        reg[0] = self.lam
        A = A + xp.diag(reg)[None]
        coef = xp.linalg.solve(A.astype(xp.float64), bvec.astype(xp.float64)[..., None])[..., 0]
        arbol.lin_feat = lf
        arbol.lin_coef = (coef * self.tasa).astype(xp.float32)

    # ------------------------------------------------------------------ pérdida
    def _grad(self, F, y, w, rng_np):
        if self.perdida == "logistica":
            p = 1 / (1 + xp.exp(-F))
            return (p - y) * w, xp.maximum(p * (1 - p), 1e-6) * w
        if self.perdida == "natural":  # #30 gradiente natural de Bernoulli: g / información de Fisher
            p = 1 / (1 + xp.exp(-F))
            gn = xp.clip((p - y) / xp.maximum(p * (1 - p), 1e-3), -20, 20)
            return gn * w, w.copy()
        if self.perdida == "focal":  # #19 diferencias centrales de la pérdida focal
            s = xp.where(y == 1, 1.0, -1.0).astype(xp.float32)
            gf = self.gamma_focal

            def L(Fx):
                lp = _logsig(s * Fx)
                return -((1 - xp.exp(lp)) ** gf) * lp
            e = 1e-2
            a, b, c = L(F + e), L(F), L(F - e)
            return (a - c) / (2 * e) * w, xp.maximum((a - 2 * b + c) / (e * e), 1e-6) * w
        if self.perdida == "pares":  # #16 logística por pares (pos, neg muestreado)
            P, N = self._pos, self._neg
            k = self.pares_k
            i = xp.repeat(P, k)
            j = N[(xp.asarray(rng_np.random(len(P) * k)) * len(N)).astype(xp.int64)]
            d = F[i] - F[j]
            s = 1 / (1 + xp.exp(d))
            hh = xp.maximum(s * (1 - s), 1e-6)
            n = len(F)
            g = suma_dispersa(n, i, (-s).astype(xp.float32)) + suma_dispersa(n, j, s.astype(xp.float32))
            h = suma_dispersa(n, i, hh.astype(xp.float32)) + suma_dispersa(n, j, hh.astype(xp.float32))
            return g / k, h / k
        raise ValueError(self.perdida)

    def _perdida_log(self, F, y, w):
        return float((w * xp.logaddexp(0, -xp.where(y == 1, F, -F))).sum())

    def _paso_lineal(self, F, v, y, w):
        """Búsqueda lineal de Newton en 1D sobre la pérdida logística (#30)."""
        r = 1.0
        for _ in range(4):
            p = 1 / (1 + xp.exp(-(F + r * v)))
            g = float((w * (p - y) * v).sum())
            h = float((w * p * (1 - p) * v * v).sum()) + 1e-9
            r = float(np.clip(r - g / h, 0.0, 10.0))
        return r

    # ------------------------------------------------------------------ ajuste
    def _salida(self, a, Xb, Z=None, Phi=None):
        if isinstance(a, Nucleo):
            return (Phi @ a.coef) * a.peso
        pos = a.rutear(_vista(Xb, a.vista))
        return a.valor_filas(pos, Z) if a.lin_feat is not None else a.valor[pos] * a.peso

    def _elegir_pares(self, Xb, g, h, rng_np, imp):
        """GA2M: entre las 20 variables con más ganancia en la fase de efectos principales, los pares
        cuyo árbol de profundidad 2 restringido al par da más ganancia sobre el gradiente actual."""
        top = np.argsort(-imp)[:20]
        cands = [(int(a), int(b)) for i, a in enumerate(top) for b in top[i + 1:]]
        idx = xp.arange(Xb.shape[0])
        gan = []
        for a, b in cands:
            arb, _ = self._crecer(Xb, g, h, idx, 2, rng_np, np.array(sorted((a, b))))
            gan.append(arb.ganancia)
        orden = np.argsort(-np.array(gan))[: self.pares_ga2m]
        return [np.array(sorted(cands[i])) for i in orden]

    def fit(self, Xb, y, Xb_val=None, y_val=None, Z=None, Z_val=None, Phi=None, Phi_val=None, F0=None, F0_val=None):
        rng = xp.random.default_rng(self.semilla) if GPU else np.random.default_rng(self.semilla)
        self._rng_gpu = rng
        rng_np = np.random.default_rng(self.semilla)
        vistas = isinstance(Xb, (list, tuple))
        Xb = [xp.asarray(v) for v in Xb] if vistas else xp.asarray(Xb)
        X0 = _vista(Xb, 0)
        y = xp.asarray(y).astype(xp.float32)
        n, F = X0.shape
        p0 = float(y.mean())
        self.base = math.log(p0 / (1 - p0)) if self.perdida != "pares" else 0.0
        Fx = xp.full(n, self.base, dtype=xp.float32)
        hay_val = Xb_val is not None
        Fv = None
        if hay_val:
            Xb_val = [xp.asarray(v) for v in Xb_val] if vistas else xp.asarray(Xb_val)
            Fv = xp.full(_vista(Xb_val, 0).shape[0], self.base, dtype=xp.float32)
        if F0 is not None:  # margen inicial (cascada local, #14)
            Fx = xp.asarray(F0, dtype=xp.float32).copy()
            if hay_val:
                Fv = xp.asarray(F0_val, dtype=xp.float32).copy()
        if self.perdida == "pares":
            self._pos = xp.nonzero(y == 1)[0]
            self._neg = xp.nonzero(y == 0)[0]
        if Phi is not None:
            Phi = xp.asarray(Phi, dtype=xp.float32)
            Phi_val = xp.asarray(Phi_val, dtype=xp.float32) if Phi_val is not None else None
        self.arboles = []
        self.historial = []
        mejor, mejor_it, sin_mejora = -1, 0, 0
        w = xp.where(y == 1, self.peso_pos, 1.0).astype(xp.float32)
        hojas_tr, hojas_va = [], []  # para DART
        orden_vars = rng_np.permutation(F)
        imp = np.zeros(F)
        pares = None
        mu = self.mu0
        if self.paso == "nesterov":
            Gx = Fx.copy()
            Gv = Fv.copy() if hay_val else None
            cF, cG = [], []
            lam_act = 1.0
        if self.paso == "infinito":
            M = xp.zeros(n, xp.float32)
            Mv = xp.zeros_like(Fv) if hay_val else None
            c = 1.0
        for it in range(self.n_rondas):
            # --- DART: abandonar árboles
            caidos = []
            if self.dart_tasa > 0 and self.arboles:
                caidos = [i for i in range(len(self.arboles)) if rng_np.random() < self.dart_tasa]
                if not caidos:
                    caidos = [int(rng_np.integers(len(self.arboles)))]
                Fd = Fx.copy()
                for i in caidos:
                    Fd -= self.arboles[i].valor[hojas_tr[i]] * self.arboles[i].peso
            else:
                Fd = Gx if self.paso == "nesterov" else Fx
            g, h = self._grad(Fd, y, w, rng_np)
            idx, mult = self._muestra(rng, g, h, n)
            gs, hs = (g, h) if mult is None else (g.copy(), h.copy())
            if mult is not None:
                gs[idx] = g[idx] * mult
                hs[idx] = h[idx] * mult
            prof = self.max_depth if not self.prof_min else int(rng_np.integers(self.prof_min, self.max_depth + 1))
            permitidas = None
            fase_pares = self.una_variable and self.pares_ga2m > 0 and it >= self.rondas_main
            if self.una_variable and not fase_pares:
                permitidas = np.array([orden_vars[it % F]])
            if fase_pares:
                if pares is None:
                    pares = self._elegir_pares(X0, g, h, rng_np, imp)
                permitidas = pares[it % len(pares)]
            k_v = int(rng_np.integers(len(Xb))) if vistas else 0
            Xc = _vista(Xb, k_v)
            self._lam_extra = mu if self.paso == "confianza" else 0.0
            arbol, pos = self._crecer(Xc, gs, hs, idx, prof, rng_np, permitidas)
            arbol.vista = k_v
            if self.una_variable and not fase_pares:
                imp[permitidas[0]] += arbol.ganancia
            if self.hojas_lineales:
                self._hojas_lineales(arbol, pos, idx, Z, gs, hs)
            aprendiz = arbol
            # --- #5 KTBoost: ¿núcleo o árbol? (el de mayor reducción de la pérdida de Newton)
            if Phi is not None:
                Pi = Phi[idx]
                A = Pi.T @ (Pi * hs[idx][:, None]) + self.nucleo_lam * xp.eye(Phi.shape[1], dtype=xp.float32)
                bvec = -(Pi.T @ gs[idx])
                beta = xp.linalg.solve(A.astype(xp.float64), bvec.astype(xp.float64))
                gan_k = 0.5 * float(bvec.astype(xp.float64) @ beta)
                if gan_k > arbol.ganancia:
                    aprendiz = Nucleo((beta * self.tasa).astype(xp.float32))
            # --- #13 InfiniteBoost: el árbol estima (F - base + paso de Newton) / c
            if self.paso == "infinito" and aprendiz is arbol:
                m_ = len(arbol.hoja)
                Fbar = suma_dispersa(m_, pos, hs[idx] * (Fx[idx] - self.base)) / xp.maximum(arbol._H, 1e-9)
                delta = -_umbral_suave(arbol._G, self.alpha) / (arbol._H + self.lam)
                arbol.valor = ((Fbar + delta) / c).astype(xp.float32)
            if isinstance(aprendiz, Nucleo):
                out_tr = Phi @ aprendiz.coef
            else:
                pos_tr = arbol.rutear(Xc)
                out_tr = arbol.valor_filas(pos_tr, Z) if self.hojas_lineales else arbol.valor[pos_tr]
            # --- #30 NGBoost: búsqueda lineal del paso
            if self.perdida == "natural":
                r = self._paso_lineal(Fx, out_tr, y, w)
                if isinstance(aprendiz, Nucleo):
                    aprendiz.coef *= r
                else:
                    arbol.valor *= r
                out_tr = out_tr * r
            # --- #17 TRBoost: cociente reducción real / reducción predicha
            if self.paso == "confianza":
                pred = -float((g * out_tr + 0.5 * h * out_tr * out_tr).sum())
                real = self._perdida_log(Fx, y, w) - self._perdida_log(Fx + out_tr, y, w)
                rho = real / pred if pred > 0 else -1.0
                mu = mu * 2.0 if rho < 0.25 else (mu / 2.0 if rho > 0.75 else mu)
                mu = float(np.clip(mu, 1e-4, 1e6))
                if rho < 0:
                    continue
            if self.dart_tasa > 0 and caidos:
                kd = len(caidos)
                arbol.peso = 1.0 / (kd + 1)
                for i in caidos:
                    viejo = self.arboles[i].peso
                    nuevo = viejo * kd / (kd + 1)
                    Fx += self.arboles[i].valor[hojas_tr[i]] * (nuevo - viejo)
                    if hay_val:
                        Fv += self.arboles[i].valor[hojas_va[i]] * (nuevo - viejo)
                    self.arboles[i].peso = nuevo
                out_tr = out_tr * arbol.peso
            out_va = None
            if hay_val:
                if isinstance(aprendiz, Nucleo):
                    out_va = Phi_val @ aprendiz.coef
                else:
                    pos_va = arbol.rutear(_vista(Xb_val, k_v))
                    out_va = arbol.valor_filas(pos_va, Z_val) if self.hojas_lineales else arbol.valor[pos_va] * arbol.peso
            # --- actualizar márgenes
            if self.paso == "nesterov":
                lam_sig = (1 + math.sqrt(1 + 4 * lam_act * lam_act)) / 2
                gam = (1 - lam_act) / lam_sig
                F_viejo = Fx
                Fx = Gx + out_tr
                Gx = (1 - gam) * Fx + gam * F_viejo
                cF_viejo = cF + [0.0]
                cF = cG + [1.0]
                cG = [(1 - gam) * a_ + gam * b_ for a_, b_ in zip(cF, cF_viejo)]
                if hay_val:
                    Fv_viejo = Fv
                    Fv = Gv + out_va
                    Gv = (1 - gam) * Fv + gam * Fv_viejo
                lam_act = lam_sig
            elif self.paso == "infinito":
                t = len(self.arboles) + 1
                M = M + (out_tr - M) / t
                for _ in range(2):  # paso de Newton amortiguado sobre la capacidad c
                    p = 1 / (1 + xp.exp(-(self.base + c * M)))
                    gc = float((w * (p - y) * M).sum())
                    hc = float((w * p * (1 - p) * M * M).sum()) + 1e-9
                    c = float(np.clip(c - self.tasa_c * gc / hc, 1e-3, 1e4))
                Fx = self.base + c * M
                if hay_val:
                    Mv = Mv + (out_va - Mv) / t
                    Fv = self.base + c * Mv
            else:
                Fx = Fx + out_tr
                if hay_val:
                    Fv = Fv + out_va
            if self.dart_tasa > 0:
                hojas_tr.append(pos_tr.astype(xp.int16))
                if hay_val:
                    hojas_va.append(pos_va.astype(xp.int16))
            self.arboles.append(aprendiz)
            # --- #10 RGF: paso totalmente correctivo sobre las hojas de todos los árboles
            if self.fc_cada > 0 and (len(self.arboles) % self.fc_cada == 0):
                for a in self.arboles:
                    if isinstance(a, Nucleo):
                        continue
                    pt = a.rutear(_vista(Xb, a.vista))
                    gg, hh = self._grad(Fx, y, w, rng_np)
                    m_ = len(a.hoja)
                    d = -self.tasa_fc * _umbral_suave(suma_dispersa(m_, pt, gg), self.alpha) / (suma_dispersa(m_, pt, hh) + self.lam)
                    d = d.astype(xp.float32)
                    a.valor = a.valor + d
                    Fx = Fx + d[pt]
                    if hay_val:
                        Fv = Fv + d[a.rutear(_vista(Xb_val, a.vista))]
            if hay_val and ((it + 1) % self.eval_cada == 0 or it == self.n_rondas - 1):
                a = auc(y_val, Fv)
                self.historial.append((it + 1, a))
                if self.verbose and (it + 1) % 100 == 0:
                    print(f"  it {it + 1} auc_val {a:.5f}", flush=True)
                if a > mejor:
                    mejor, mejor_it, sin_mejora = a, it + 1, 0
                elif not (self.una_variable and self.pares_ga2m > 0 and it < self.rondas_main):
                    sin_mejora += self.eval_cada
                    if sin_mejora >= self.paciencia:
                        break
        for a in self.arboles:
            if hasattr(a, "_G"):
                del a._G, a._H
        if self.paso == "nesterov":
            for a, cf in zip(self.arboles, cF):
                a.peso = a.peso * cf
        if self.paso == "infinito":
            T = len(self.arboles)
            for a in self.arboles:
                a.peso = c / T
            self.c = c
        self.mejor_auc, self.mejor_it = mejor, mejor_it
        especial = self.dart_tasa > 0 or self.paso in ("nesterov", "infinito", "confianza") or self.fc_cada > 0
        if hay_val and mejor_it and not especial:
            self.arboles = self.arboles[:mejor_it]
        return self

    def decision(self, Xb, Z=None, Phi=None, F0=None):
        vistas = isinstance(Xb, (list, tuple))
        Xb = [xp.asarray(v) for v in Xb] if vistas else xp.asarray(Xb)
        if Phi is not None:
            Phi = xp.asarray(Phi, dtype=xp.float32)
        Fx = xp.full(_vista(Xb, 0).shape[0], self.base, dtype=xp.float32)
        if F0 is not None:
            Fx = xp.asarray(F0, dtype=xp.float32).copy()
        for a in self.arboles:
            Fx += self._salida(a, Xb, Z, Phi)
        return Fx
