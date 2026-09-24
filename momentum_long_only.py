"""
Momentum LONG-ONLY (señal lookback-skip configurable: 12-1 por defecto, --lookback 9 para 9-1) sobre el universo B3 del listado, medido contra el IBOV y contra el propio universo.
Reutiliza datos, limpieza, filtro de elegibilidad y señal de momentum_brasil.py.

Uso:
    python momentum_long_only.py --prices "datos_b3 1.xlsx" --factors "factor_bra.xlsm" --cost-bps 30

Estrategias evaluadas (rebalanceo mensual, formación en t, tenencia t+1):
  UNIV_vw / UNIV_ew / UNIV_vwcap : universo elegible sin señal (vw tope 20%, equal-weight, capped vw)
                 -> benchmarks "like-for-like" para aislar el aporte del momentum
  TOP_T_vwcap  : tercil ganador, capped value-weight (JKP, p80)
  TOP_T_ew     : tercil ganador, equal-weight
  TOP_T_buf    : tercil ganador vw_cap con buffer (entra en top 33%, sale si cae debajo del top 50%)
  TILT_l       : universo vw con tilt multiplicativo w_i = w_b,i * max(0, 1 + l * z_i), z = z-score del momentum
                 (cartera cercana al benchmark: controla tracking error con l)

Métricas: retorno y exceso vs IBOV y vs universo, tracking error, IR, t-stat Newey-West del exceso,
beta vs IBOV, max drawdown relativo, turnover y exceso neto de costos (cost_bps por lado sobre el turnover).
Regresión del exceso vs IBOV sobre factores JKP (USD) para ver si el alfa es momentum "de libro".
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

import momentum_brasil as mb


# ----------------------------------------------------------------------------
# PESOS
# ----------------------------------------------------------------------------

def normalize(w: pd.DataFrame) -> pd.DataFrame:
    return w.div(w.sum(axis=1).replace(0, np.nan), axis=0)


def cap_single_name(w: pd.DataFrame, max_w: float = 0.20, iters: int = 20) -> pd.DataFrame:
    """Tope por nombre con redistribución proporcional (estilo índice)."""
    w = normalize(w)
    for _ in range(iters):
        over = w > max_w
        if not over.any().any():
            break
        excess = (w - max_w).clip(lower=0).sum(axis=1)
        w = w.where(~over, max_w)
        free = w.where(~over)
        w = w.where(over, free.add(free.div(free.sum(axis=1), axis=0).mul(excess, axis=0), fill_value=0))
    return w


def benchmark_weights(mcap, eligible, max_w=0.20):
    return cap_single_name(mcap.where(eligible), max_w)


def top_group_weights(score, mcap, eligible, n_groups=3, weighting="vw_cap", cap_pct=0.8, min_stocks=25):
    g = mb.assign_groups(score.where(eligible), n_groups, min_stocks)
    base = mb.formation_weights(mcap.where(eligible), weighting, cap_pct)
    return normalize(base.where(g == n_groups))


def buffered_top_weights(score, mcap, eligible, entry=2 / 3, exit_=0.5, weighting="vw_cap", cap_pct=0.8, min_stocks=25):
    """Histeresis: entra si el percentil de momentum >= entry; se mantiene mientras percentil >= exit_."""
    pct = score.where(eligible).rank(axis=1, pct=True)
    enough = score.where(eligible).notna().sum(axis=1) >= min_stocks
    held = pd.DataFrame(False, index=pct.index, columns=pct.columns)
    prev = pd.Series(False, index=pct.columns)
    for t in pct.index:
        if not enough.loc[t]:
            prev = pd.Series(False, index=pct.columns); continue
        p = pct.loc[t]
        cur = (p >= entry) | (prev & (p >= exit_))
        held.loc[t] = cur.fillna(False)
        prev = held.loc[t]
    base = mb.formation_weights(mcap.where(eligible), weighting, cap_pct)
    return normalize(base.where(held))


def tilt_weights(score, wb, lam=0.5, clip_z=3.0):
    """w = w_b * max(0, 1 + lam * z); z-score cross-sectional del momentum dentro del universo elegible."""
    s = score.where(wb.notna())
    z = s.sub(s.mean(axis=1), axis=0).div(s.std(axis=1), axis=0).clip(-clip_z, clip_z)
    return normalize(wb * (1 + lam * z).clip(lower=0))


# ----------------------------------------------------------------------------
# RETORNOS Y MÉTRICAS
# ----------------------------------------------------------------------------

def port_return(w: pd.DataFrame, mret: pd.DataFrame) -> pd.Series:
    """Pesos de t aplicados al retorno de t+1 (renormalizados si falta algún retorno)."""
    wl = w.shift(1).where(mret.notna())
    return (normalize(wl) * mret).sum(axis=1, min_count=1)


def nw_tstat(x: pd.Series, lags: int = 6) -> float:
    x = x.dropna()
    r = sm.OLS(x.values, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(r.tvalues[0])


def relative_stats(r: pd.Series, b: pd.Series, to: pd.Series | None = None, cost_bps: float = 30) -> pd.Series:
    d = pd.concat([r.rename("r"), b.rename("b")], axis=1).dropna()
    ex = d.r - d.b
    te = ex.std() * np.sqrt(12)
    rel = (1 + d.r).cumprod() / (1 + d.b).cumprod()
    beta = np.cov(d.r, d.b)[0, 1] / d.b.var()
    out = {
        "Ret. anualizado": (1 + d.r).prod() ** (12 / len(d)) - 1,
        "Bench anualizado": (1 + d.b).prod() ** (12 / len(d)) - 1,
        "Exceso geom. anual": rel.iloc[-1] ** (12 / len(d)) - 1,
        "Exceso aritm. anual": ex.mean() * 12,
        "Tracking error": te,
        "IR": ex.mean() * 12 / te if te > 0 else np.nan,
        "t exceso (NW)": nw_tstat(ex),
        "Beta vs bench": beta,
        "Hit ratio mensual": (ex > 0).mean(),
        "Max DD relativo": (rel / rel.cummax() - 1).min(),
        "Vol. absoluta": d.r.std() * np.sqrt(12),
        "Max DD absoluto": ((1 + d.r).cumprod() / (1 + d.r).cumprod().cummax() - 1).min(),
        "N meses": len(d),
    }
    if to is not None:
        tc = to.reindex(d.index).fillna(0) * 2 * cost_bps / 1e4   # compra + venta del turnover one-way
        exn = ex - tc
        out.update({"Turnover one-way mensual": to.reindex(d.index).mean(),
                    f"Exceso aritm. neto ({cost_bps:.0f}bps)": exn.mean() * 12,
                    "IR neto": exn.mean() * 12 / te if te > 0 else np.nan,
                    "t neto (NW)": nw_tstat(exn)})
    return pd.Series(out)


def rolling_excess(r, b, window=36):
    ex = (r - b).dropna()
    return ex.rolling(window).mean() * 12


# ----------------------------------------------------------------------------
# PIPELINE
# ----------------------------------------------------------------------------

def run(prices_path, factors_path, currency="usd", cost_bps=30.0, max_w=0.20, lambdas=(0.25, 0.5, 1.0),
        out_dir="resultados_long_only", hac_lags=6, lookback=12, skip=1):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    S = mb.build_strategy(prices_path, factors_path, lookback=lookback, skip=skip, currency=currency, verbose=False)
    score, mcap, el, mret = S["score"], S["mcap"], S["eligible"], S["mret"]
    ibov = S["port"]["MKT_IBOV"]

    W = {"UNIV_vw": benchmark_weights(mcap, el, max_w),
         "UNIV_ew": normalize(el.astype(float).where(el)),
         "UNIV_vwcap": normalize(mb.formation_weights(mcap.where(el), "vw_cap"))}
    W["TOP_T_vwcap"] = top_group_weights(score, mcap, el, 3, "vw_cap")
    W["TOP_T_ew"] = top_group_weights(score, mcap, el, 3, "ew")
    W["TOP_T_buf"] = buffered_top_weights(score, mcap, el)
    W["TOP_T_buf_ew"] = buffered_top_weights(score, mcap, el, weighting="ew")
    for l in lambdas:
        W[f"TILT_{l}"] = tilt_weights(score, W["UNIV_vw"], l)

    R = pd.DataFrame({k: port_return(w, mret) for k, w in W.items()})
    R["IBOV"] = ibov
    R = R.loc[S["port"].index].dropna()
    TO = pd.DataFrame({k: mb.turnover(w, mret) for k, w in W.items()}).reindex(R.index)
    NN = pd.DataFrame({k: w.notna().sum(axis=1) for k, w in W.items()}).reindex(R.index)

    strategies = [k for k in W]
    vs_ibov = pd.DataFrame({k: relative_stats(R[k], R["IBOV"], TO[k], cost_bps) for k in strategies})
    # Comparación "like-for-like": cada estrategia contra el universo elegible con su MISMA ponderación.
    # Aísla el aporte de la señal del sesgo de universo (la lista actual le gana al IBOV sin señal) y del sesgo de tamaño.
    own = {"TOP_T_vwcap": "UNIV_vwcap", "TOP_T_buf": "UNIV_vwcap", "TOP_T_ew": "UNIV_ew", "TOP_T_buf_ew": "UNIV_ew"}
    own.update({k: "UNIV_vw" for k in strategies if k.startswith("TILT")})
    vs_univ = pd.DataFrame({f"{k} vs {b}": relative_stats(R[k], R[b], TO[k], cost_bps) for k, b in own.items()})

    # Subperíodos (exceso aritm. anual vs IBOV)
    bins = pd.PeriodIndex(R.index)
    lab = pd.Series([f"{y//5*5}-{min(y//5*5+4, bins.year.max())}" for y in bins.year], index=R.index)
    sub = (R[strategies].sub(R["IBOV"], axis=0)).groupby(lab).mean() * 12

    # Regresiones del exceso vs IBOV sobre factores JKP (USD)
    fac = mb.load_factors(factors_path)
    jkp = mb.load_jkp_momentum(factors_path, col=f"ret_{lookback}_{skip}")
    if jkp is None:
        jkp = mb.load_jkp_momentum(factors_path)
    ctrl = [c for c in fac.columns if c != "momentum"]
    regs = {}
    for k in ["TOP_T_vwcap", "TOP_T_ew", "TILT_0.5"]:
        ex = (R[k] - R["IBOV"]).rename(k)
        regs[f"{k}_CAPM"] = mb.ols_alpha(R[k], R[["IBOV"]], hac_lags)
        regs[f"{k}_vs_JKP_mom"] = mb.ols_alpha(ex, jkp.to_frame(), hac_lags)
        regs[f"{k}_vs_clusters"] = mb.ols_alpha(ex, fac[ctrl].join(jkp), hac_lags)

    with pd.ExcelWriter(out / "resultados_long_only.xlsx") as xw:
        vs_ibov.to_excel(xw, sheet_name="vs_IBOV")
        vs_univ.to_excel(xw, sheet_name="vs_universo_misma_pond")
        sub.to_excel(xw, sheet_name="exceso_subperiodos")
        R.to_excel(xw, sheet_name="retornos_mensuales")
        TO.to_excel(xw, sheet_name="turnover")
        NN.to_excel(xw, sheet_name="n_nombres")
        for k, t in regs.items():
            t.to_excel(xw, sheet_name=k[:31])
        for k in ["TOP_T_vwcap", "TILT_0.5"]:
            W[k].dropna(how="all", axis=1).loc[R.index.min() - 1:].to_excel(xw, sheet_name=f"pesos_{k}"[:31])
        # cartera vigente (formada al último cierre disponible)
        last = score.dropna(how="all").index.max()
        cur = pd.DataFrame({k: W[k].loc[last] for k in ["UNIV_vw", "TOP_T_vwcap", "TOP_T_ew", "TILT_0.5"]})
        cur["score_12_1"] = score.loc[last]
        cur.dropna(how="all", subset=["UNIV_vw", "TOP_T_vwcap"]).sort_values("score_12_1", ascending=False) \
            .to_excel(xw, sheet_name=f"cartera_{last}")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        idx = R.index.to_timestamp()
        for k in ["TOP_T_vwcap", "TOP_T_ew", "TILT_0.5", "UNIV_ew", "UNIV_vw"]:
            ax[0].plot(idx, ((1 + R[k]).cumprod() / (1 + R["IBOV"]).cumprod()).values, label=k)
            ax[1].plot(idx, rolling_excess(R[k], R["IBOV"]).values, label=k)
        ax[0].axhline(1, color="grey", lw=.8); ax[0].set_title("Riqueza relativa vs IBOV"); ax[0].legend()
        ax[1].axhline(0, color="grey", lw=.8); ax[1].set_title("Exceso anualizado vs IBOV, ventana 36m")
        plt.tight_layout(); plt.savefig(out / "long_only_vs_ibov.png", dpi=130); plt.close()
    except Exception as e:
        print("Gráfico omitido:", e)

    pd.options.display.float_format = "{:,.3f}".format
    print(f"\nLong-only momentum {lookback}-{skip} | {currency.upper()} | {R.index.min()} a {R.index.max()} ({len(R)} meses)"
          f" | costos {cost_bps:.0f} bps por lado\nNombres medios: {NN.mean().round(1).to_dict()}")
    print("\n=== vs IBOV ===\n", vs_ibov)
    print("\n=== vs universo elegible con la misma ponderación (aporte de la señal) ===\n", vs_univ.T)
    print("\n=== Exceso anual vs IBOV por subperíodo ===\n", sub)
    for k, t in regs.items():
        print(f"\n--- {k} | R2={t.attrs['r2']:.3f} N={t.attrs['nobs']} ---\n", t[["coef", "t (NW)", "alfa anual"]])
    return dict(R=R, W=W, vs_ibov=vs_ibov, vs_univ=vs_univ, sub=sub, regs=regs, TO=TO)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", required=True)
    ap.add_argument("--factors", required=True)
    ap.add_argument("--currency", default="usd", choices=["usd", "brl"])
    ap.add_argument("--cost-bps", type=float, default=30.0)
    ap.add_argument("--max-w", type=float, default=0.20)
    ap.add_argument("--lookback", type=int, default=12)
    ap.add_argument("--skip", type=int, default=1)
    ap.add_argument("--out", default="resultados_long_only")
    a = ap.parse_args()
    run(a.prices, a.factors, a.currency, a.cost_bps, a.max_w, out_dir=a.out, lookback=a.lookback, skip=a.skip)
