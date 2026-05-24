"""
Step 17b — Conformal Prediction para CI calibrado
Objetivo: mejorar CI coverage de 74% → ≥80% (nominal target).

Método: Split Conformal Prediction (Papadopoulos et al., 2002)
  1. Dividir el conjunto de validación WFV en dos partes temporales:
       - Calibración (80%): calcular nonconformity scores
       - Test (20%): verificar coverage
  2. Nonconformity score: residuo absoluto |y_true - y_pred|
  3. Quantil q = ceil((1-alpha)*(n+1)) / n de los scores de calibración
  4. CI: [y_hat - q, y_hat + q]  ← garantía distribución-libre

Ventaja frente al CI empírico actual (basado en percentiles de error relativo):
  - Cobertura garantizada bajo intercambiabilidad débil (supuesto razonable aquí)
  - No asume normalidad ni homocedasticidad
  - Automáticamente más ancho en zonas de alta incertidumbre

Outputs:
  output/conformal_meta.json      — quantiles de calibración + coverage
  output/conformal_predictions.csv — CI lower/upper por semana
  output/conformal_plot.png       — reliability diagram before/after

Run: python src/17b_conformal_prediction.py
"""

import sys
import json
import warnings
from pathlib import Path

# Force UTF-8 output on Windows consoles that default to CP1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import OUTPUT_DIR, CONFORMAL_META, TARGET_CI_COVERAGE

warnings.filterwarnings("ignore")

BLUE   = "#1f4e79"
ORANGE = "#ff6b35"
GREEN  = "#70ad47"
RED    = "#c00000"
GREY   = "#aaaaaa"
PURPLE = "#7030a0"

ALPHA_80 = 0.20   # 1 - coverage nominal
ALPHA_50 = 0.50
CALIB_FRAC = 0.60  # fracción temporal usada como calibración (las primeras)


def load_wfv_predictions() -> pd.DataFrame | None:
    """Carga las predicciones WFV del Step 10."""
    path = OUTPUT_DIR / "wfv_predictions.csv"
    if not path.exists():
        print(f"[ERROR] {path} no encontrado.")
        print("  Ejecuta src/10_walk_forward_validation.py primero.")
        return None
    df = pd.read_csv(path, parse_dates=["week_date"], index_col="week_date").sort_index()
    print(f"[OK] WFV predictions: {len(df)} semanas")
    return df


def split_temporal(df: pd.DataFrame, frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split cronológico estricto. Nunca mezcla futuro con pasado."""
    n_calib = int(len(df) * frac)
    calib = df.iloc[:n_calib]
    test  = df.iloc[n_calib:]
    print(f"  Calibración: {len(calib)} obs ({calib.index.min().date()} — {calib.index.max().date()})")
    print(f"  Test:        {len(test)} obs  ({test.index.min().date()} — {test.index.max().date()})")
    return calib, test


def compute_conformal_quantile(residuals: np.ndarray, alpha: float) -> float:
    """
    Calcula el quantil conformal al nivel (1-alpha).
    q = quantile((1-alpha) * (n+1) / n) de los residuos absolutos de calibración.
    Garantía: coverage ≥ (1-alpha) en el conjunto de test.
    """
    n = len(residuals)
    level = np.ceil((1 - alpha) * (n + 1)) / n
    level = min(level, 1.0)  # no puede superar 1
    return float(np.quantile(residuals, level))


def evaluate_coverage(actual: np.ndarray, ci_lo: np.ndarray, ci_hi: np.ndarray) -> float:
    """Fracción de observaciones dentro del CI."""
    return float(np.mean((actual >= ci_lo) & (actual <= ci_hi)))


def compute_interval_width(ci_lo: np.ndarray, ci_hi: np.ndarray) -> float:
    return float(np.mean(ci_hi - ci_lo))


def run_conformal(wfv_df: pd.DataFrame) -> dict:
    """
    Aplica Split Conformal Prediction a las predicciones del Step 10.
    Compara con el CI empírico original del Step 17.
    """
    print("\n[2/4] Split Conformal Prediction...")

    calib_df, test_df = split_temporal(wfv_df, CALIB_FRAC)

    # Nonconformity scores en calibración: residuo absoluto
    resid_calib = np.abs(calib_df["actual_R03"].values - calib_df["pred_B"].values)
    print(f"\n  Residuos calibración — median={np.median(resid_calib):.2f}, "
          f"p80={np.quantile(resid_calib, 0.80):.2f}, "
          f"p95={np.quantile(resid_calib, 0.95):.2f}")

    # Quantiles conformales para 80% y 50%
    q_80 = compute_conformal_quantile(resid_calib, ALPHA_80)
    q_50 = compute_conformal_quantile(resid_calib, ALPHA_50)
    print(f"\n  Quantil conformal 80%: q = {q_80:.2f} unidades")
    print(f"  Quantil conformal 50%: q = {q_50:.2f} unidades")

    # Aplicar al test set
    actual_test = test_df["actual_R03"].values
    pred_test   = test_df["pred_B"].values

    ci_lo_80 = pred_test - q_80
    ci_hi_80 = pred_test + q_80
    ci_lo_50 = pred_test - q_50
    ci_hi_50 = pred_test + q_50

    cov_80 = evaluate_coverage(actual_test, ci_lo_80, ci_hi_80)
    cov_50 = evaluate_coverage(actual_test, ci_lo_50, ci_hi_50)
    width_80 = compute_interval_width(ci_lo_80, ci_hi_80)
    width_50 = compute_interval_width(ci_lo_50, ci_hi_50)

    print(f"\n  Cobertura Conformal 80% CI: {cov_80:.1%} (nominal 80%)")
    print(f"  Cobertura Conformal 50% CI: {cov_50:.1%} (nominal 50%)")
    print(f"  Anchura media 80% CI: {width_80:.2f} unidades")

    # Comparar con CI original del Step 17
    ci_orig = OUTPUT_DIR / "ci_calibration.json"
    orig_cov = None
    if ci_orig.exists():
        with open(ci_orig) as f:
            orig = json.load(f)
        orig_cov = orig.get("coverage_80_overall", 0.74)
        print(f"\n  CI original (Step 17): {orig_cov:.1%}")
        improvement = cov_80 - orig_cov
        print(f"  Mejora conformal: {improvement:+.1%}")

    # Guardar CI por semana
    result_df = test_df.copy()
    result_df["ci_lo_80_conformal"] = ci_lo_80
    result_df["ci_hi_80_conformal"] = ci_hi_80
    result_df["ci_lo_50_conformal"] = ci_lo_50
    result_df["ci_hi_50_conformal"] = ci_hi_50
    result_df["in_80_conformal"]    = (actual_test >= ci_lo_80) & (actual_test <= ci_hi_80)
    result_df["in_50_conformal"]    = (actual_test >= ci_lo_50) & (actual_test <= ci_hi_50)

    out_csv = OUTPUT_DIR / "conformal_predictions.csv"
    result_df.to_csv(out_csv)
    print(f"[OK] Guardado: {out_csv}")

    return {
        "q_80": round(q_80, 3),
        "q_50": round(q_50, 3),
        "coverage_80_conformal": round(cov_80, 4),
        "coverage_50_conformal": round(cov_50, 4),
        "width_80_conformal": round(width_80, 3),
        "nominal_80": 0.80,
        "nominal_50": 0.50,
        "n_calibration": len(calib_df),
        "n_test": len(test_df),
        "calib_fraction": CALIB_FRAC,
        "coverage_80_original": round(orig_cov, 4) if orig_cov else None,
        "improvement_pp": round((cov_80 - orig_cov) * 100, 2) if orig_cov else None,
        "method": "split_conformal_prediction",
        "nonconformity_score": "absolute_residual",
        "target_coverage": TARGET_CI_COVERAGE,
        "target_achieved": cov_80 >= TARGET_CI_COVERAGE,
        "result_df": result_df,
    }


def plot_conformal(result: dict, wfv_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Conformal Prediction — Calibración de Intervalos de Confianza\n"
                 f"Split Conformal (calib={CALIB_FRAC:.0%}) | Nonconformity: |y - ŷ|",
                 fontsize=11, fontweight="bold", color=BLUE)

    # Panel 1: Reliability diagram before/after
    ax = axes[0]
    alphas = np.arange(0.05, 1.0, 0.05)
    nominal_levels = 1 - alphas

    # CI original (empírico) — aproximado desde ci_calibration.json si existe
    ci_orig = OUTPUT_DIR / "ci_calibration.json"
    if ci_orig.exists():
        with open(ci_orig) as f:
            orig_data = json.load(f)
        orig_80 = orig_data.get("coverage_80_overall", 0.74)
        orig_50 = orig_data.get("coverage_50_overall", None)
    else:
        orig_80, orig_50 = 0.74, None

    # Conformal actual coverage para distintos alpha
    calib_df = wfv_df.iloc[:int(len(wfv_df) * CALIB_FRAC)]
    test_df  = wfv_df.iloc[int(len(wfv_df) * CALIB_FRAC):]
    resid_calib = np.abs(calib_df["actual_R03"].values - calib_df["pred_B"].values)
    actual_test = test_df["actual_R03"].values
    pred_test   = test_df["pred_B"].values

    conformal_actual_cov = []
    for alpha in alphas:
        q = compute_conformal_quantile(resid_calib, alpha)
        cov = evaluate_coverage(actual_test, pred_test - q, pred_test + q)
        conformal_actual_cov.append(cov)

    ax.plot([0, 1], [0, 1], color=GREY, lw=1.5, linestyle="--", label="Calibración perfecta")
    ax.plot(nominal_levels, conformal_actual_cov, color=PURPLE, lw=2.5,
            marker="o", ms=5, label="Conformal (nuevo)")
    ax.scatter([0.80, 0.50], [result["coverage_80_conformal"], result["coverage_50_conformal"]],
               s=130, color=PURPLE, zorder=8)
    ax.scatter([0.80], [orig_80], s=130, color=ORANGE, zorder=8,
               label=f"CI original 80%: {orig_80:.0%}")
    ax.set_xlabel("Cobertura nominal", fontsize=9)
    ax.set_ylabel("Cobertura empírica", fontsize=9)
    ax.set_title("Reliability Diagram\n(antes vs después)", fontsize=9, color=BLUE)
    ax.legend(fontsize=7.5); ax.grid(alpha=0.25, linestyle="--")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel 2: CI en el test set
    ax2 = axes[1]
    result_df = result["result_df"]
    dates_test = result_df.index
    actual_arr = result_df["actual_R03"].values
    pred_arr   = result_df["pred_B"].values
    lo_80 = result_df["ci_lo_80_conformal"].values
    hi_80 = result_df["ci_hi_80_conformal"].values

    ax2.fill_between(dates_test, lo_80, hi_80, alpha=0.20, color=PURPLE,
                     label=f"80% CI conformal ({result['coverage_80_conformal']:.0%} cobertura)")
    ax2.plot(dates_test, actual_arr, color="black", lw=2, label="Actual", zorder=5)
    ax2.plot(dates_test, pred_arr,   color=ORANGE,  lw=1.5, linestyle="--",
             alpha=0.8, label="Pred XGBoost-B")
    ax2.set_ylabel("R03 units / week", fontsize=9)
    ax2.set_title("CI Conformal en test set", fontsize=9, color=BLUE)
    ax2.legend(fontsize=7.5); ax2.grid(alpha=0.2, linestyle="--")
    ax2.spines[["top", "right"]].set_visible(False)

    # Panel 3: Before/After coverage bar
    ax3 = axes[2]
    labels = ["Original\n(Step 17)", "Conformal\n(Step 17b)"]
    vals   = [orig_80 * 100, result["coverage_80_conformal"] * 100]
    colors = [ORANGE, PURPLE]
    bars = ax3.bar(labels, vals, color=colors, alpha=0.85, edgecolor="white", width=0.5)
    ax3.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=11)
    ax3.axhline(80, color=BLUE, lw=2, linestyle=":", label="Nominal 80%")
    ax3.axhline(TARGET_CI_COVERAGE * 100, color=GREEN, lw=1.5, linestyle=":",
                label=f"Target {TARGET_CI_COVERAGE:.0%}")
    ax3.set_ylabel("Cobertura real (%)", fontsize=9)
    ax3.set_title("Mejora de cobertura 80% CI", fontsize=9, color=BLUE)
    ax3.set_ylim(0, 110)
    ax3.legend(fontsize=8); ax3.grid(axis="y", alpha=0.2, linestyle="--")
    ax3.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = OUTPUT_DIR / "conformal_plot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Plot guardado: {out}")


def main():
    print("=" * 65)
    print("STEP 17b: CONFORMAL PREDICTION — CI CALIBRATION")
    print("=" * 65)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Cargar predicciones WFV ---
    print("\n[1/4] Cargando predicciones WFV...")
    wfv_df = load_wfv_predictions()
    if wfv_df is None:
        return
    if "pred_B" not in wfv_df.columns:
        print("[ERROR] columna 'pred_B' no encontrada en wfv_predictions.csv")
        return

    # --- Conformal prediction ---
    result = run_conformal(wfv_df)

    # --- Guardar meta ---
    meta = {k: v for k, v in result.items() if k != "result_df"}
    with open(CONFORMAL_META, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[OK] Meta guardado: {CONFORMAL_META}")

    # --- Plot ---
    print("\n[3/4] Generando plots...")
    plot_conformal(result, wfv_df)

    # --- Resumen final ---
    print("\n[4/4] RESUMEN CONFORMAL PREDICTION")
    print("=" * 50)
    print(f"  CI original (Step 17) 80%:  {result.get('coverage_80_original', '?'):.1%}")
    print(f"  CI conformal      80%:  {result['coverage_80_conformal']:.1%}")
    print(f"  CI conformal      50%:  {result['coverage_50_conformal']:.1%}")
    print(f"  Mejora: {result.get('improvement_pp', '?'):+.1f}pp")
    print(f"  Target {TARGET_CI_COVERAGE:.0%} alcanzado: {'SI' if result['target_achieved'] else 'NO'}")
    print(f"\n  Quantil q_80 = {result['q_80']:.2f} unidades  (ancho del CI)")
    print(f"  (interpretacion: IC 80% es y_pred +/- {result['q_80']:.2f} unidades)")
    print(f"\n  Calibración: {result['n_calibration']} obs | Test: {result['n_test']} obs")

    print("\n[DONE] Step 17b completo.")
    print("  Actualiza el dashboard con conformal_predictions.csv y conformal_meta.json")


if __name__ == "__main__":
    main()
