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
  #29 GA2M/EBM: un árbol por variable, ciclo de variables                          una_variable=True
"""
import math
import numpy as np

try:
    import cupy as cp
    import cupyx
    xp = cp
    GPU = True
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


class BoostingNewton:
    def __init__(self, n_rondas=1000, tasa=0.05, max_depth=6, prof_min=None, lam=1.0, alpha=0.0,
                 gamma=0.0, min_hess=1.0, submuestra=0.8, colsample=0.5, peso_pos=1.0,
                 hojas_lineales=False, lam_lin=10.0, dart_tasa=0.0, olvidadizo=False,
                 muestreo="bernoulli", una_variable=False, semilla=0, paciencia=100,
                 eval_cada=10, verbose=False, max_delta=0.0):
        self.__dict__.update({k: v for k, v in locals().items() if k != "self"})

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
        v = -_umbral_suave(Gh, self.alpha) / (Hh + self.lam)
        if self.max_delta > 0:
            v = xp.clip(v, -self.max_delta, self.max_delta)
        arbol.valor = (v * self.tasa).astype(xp.float32)
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

    # ------------------------------------------------------------------ ajuste
    def fit(self, Xb, y, Xb_val=None, y_val=None, Z=None, Z_val=None):
        rng = xp.random.default_rng(self.semilla) if GPU else np.random.default_rng(self.semilla)
        rng_np = np.random.default_rng(self.semilla)
        Xb = xp.asarray(Xb)
        y = xp.asarray(y).astype(xp.float32)
        n, F = Xb.shape
        p0 = float(y.mean())
        self.base = math.log(p0 / (1 - p0))
        Fx = xp.full(n, self.base, dtype=xp.float32)
        hay_val = Xb_val is not None
        if hay_val:
            Xb_val = xp.asarray(Xb_val)
            Fv = xp.full(Xb_val.shape[0], self.base, dtype=xp.float32)
        self.arboles = []
        self.historial = []
        mejor, mejor_it, sin_mejora = -1, 0, 0
        w = xp.where(y == 1, self.peso_pos, 1.0).astype(xp.float32)
        hojas_tr, hojas_va = [], []  # para DART
        orden_vars = rng_np.permutation(F)
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
                Fd = Fx
            p = 1 / (1 + xp.exp(-Fd))
            g = (p - y) * w
            h = xp.maximum(p * (1 - p), 1e-6) * w
            idx, mult = self._muestra(rng, g, h, n)
            gs, hs = (g, h) if mult is None else (g.copy(), h.copy())
            if mult is not None:
                gs[idx] = g[idx] * mult
                hs[idx] = h[idx] * mult
            prof = self.max_depth if not self.prof_min else int(rng_np.integers(self.prof_min, self.max_depth + 1))
            permitidas = None
            if self.una_variable:
                permitidas = np.array([orden_vars[it % F]])
            arbol, pos = self._crecer(Xb, gs, hs, idx, prof, rng_np, permitidas)
            if self.hojas_lineales:
                self._hojas_lineales(arbol, pos, idx, Z, gs, hs)
            pos_tr = arbol.rutear(Xb)
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
            Fx += arbol.valor_filas(pos_tr, Z) if self.hojas_lineales else arbol.valor[pos_tr] * arbol.peso
            if hay_val:
                pos_va = arbol.rutear(Xb_val)
                Fv += arbol.valor_filas(pos_va, Z_val) if self.hojas_lineales else arbol.valor[pos_va] * arbol.peso
            if self.dart_tasa > 0:
                hojas_tr.append(pos_tr.astype(xp.int16))
                if hay_val:
                    hojas_va.append(pos_va.astype(xp.int16))
            self.arboles.append(arbol)
            if hay_val and ((it + 1) % self.eval_cada == 0 or it == self.n_rondas - 1):
                a = auc(y_val, Fv)
                self.historial.append((it + 1, a))
                if self.verbose and (it + 1) % 100 == 0:
                    print(f"  it {it + 1} auc_val {a:.5f}", flush=True)
                if a > mejor:
                    mejor, mejor_it, sin_mejora = a, it + 1, 0
                else:
                    sin_mejora += self.eval_cada
                    if sin_mejora >= self.paciencia:
                        break
        self.mejor_auc, self.mejor_it = mejor, mejor_it
        if hay_val and mejor_it:
            self.arboles = self.arboles[:mejor_it] if self.dart_tasa == 0 else self.arboles
        return self

    def decision(self, Xb, Z=None):
        Xb = xp.asarray(Xb)
        Fx = xp.full(Xb.shape[0], self.base, dtype=xp.float32)
        for a in self.arboles:
            pos = a.rutear(Xb)
            Fx += a.valor_filas(pos, Z) if a.lin_feat is not None else a.valor[pos] * a.peso
        return Fx
