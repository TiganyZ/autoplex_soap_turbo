"""ir_fft.f90's estimator, in numpy, so a spectrum can be recomputed from the
dipole time series alone.

Why reimplement something turboGAP already does: a 10 ps trajectory gives ONE
spectrum and no error bar, and the question "is the new dipole model better?"
is unanswerable without one. Splitting the series into blocks and re-estimating
per block gives the spread, and doing that through turboGAP would mean 2N more
prediction runs -- when the dipoles it already wrote are the only input the
estimator needs.

Validated against turboGAP's own output:
`python -m autoplex_soap_turbo.turbogap.ir_spectrum check <dir>`
recomputes a finished run's spectrum from its ir_fft_dipoles.dat and reports
the largest relative difference from its ir_fft_spectrum.dat. Anything above
~1e-9 means this is no longer the same estimator and the error bars it
produces do not belong beside turboGAP's numbers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CM_PER_INV_FS = 33356.40952
HBAR_C_EV_CM = 1.23984e-4
KB_EV = 8.617333e-5


def autocorrelation(mu: np.ndarray, subtract_mean: bool = True) -> np.ndarray:
    """Summed over the three Cartesian components, normalised by n_frames.

    Zero-padded to the next power of two before the FFT, exactly as the Fortran
    does: a shorter pad wraps the tail of the series onto its head and the long
    lags -- the ones the window keeps -- are the ones that get it wrong.
    """
    n_frames = mu.shape[0]
    n = 1
    while n < 2 * n_frames:
        n *= 2
    acf = np.zeros(n_frames)
    means = mu.mean(axis=0) if subtract_mean else np.zeros(3)
    for d in range(3):
        x = np.zeros(n)
        x[:n_frames] = mu[:, d] - means[d]
        z = np.fft.fft(x)
        acf += np.fft.ifft(np.abs(z) ** 2).real[:n_frames]
    return acf / n_frames, means


def lag_window(kind: str, L: int) -> np.ndarray:
    t = np.arange(L, dtype=float)
    if kind in ("none", "boxcar"):
        return np.ones(L)
    if kind == "bartlett":
        return 1.0 - t / L
    if kind == "welch":
        return 1.0 - (t / L) ** 2
    # hann, as the Python it was translated from writes it
    return 0.5 * (np.cos(np.pi * t / L) + 1.0)


def prefactor(kind: str, temperature: float, nu: np.ndarray) -> np.ndarray:
    if kind in ("classical", "quadratic"):
        return nu ** 2
    if kind == "linear":
        return nu
    if kind == "none":
        return np.ones_like(nu)
    kT = max(KB_EV * temperature, 1e-30)
    return nu * (1.0 - np.exp(-HBAR_C_EV_CM * nu / kT))


def smooth_gaussian(x: np.ndarray, smooth_k: int) -> np.ndarray:
    """Edge-clamped, which is what the Fortran does and what np.convolve does
    not; at smooth_k = 10 that is 21 bins at each end."""
    sigma = smooth_k / 2.355
    radius = int(4.0 * sigma + 0.5)
    if radius < 1:
        return x.copy()
    i = np.arange(-radius, radius + 1)
    w = np.exp(-0.5 * (i / sigma) ** 2)
    w /= w.sum()
    idx = np.clip(np.arange(len(x))[:, None] + i[None, :], 0, len(x) - 1)
    return (x[idx] * w[None, :]).sum(axis=1)


def spectrum(mu: np.ndarray, dt_fs: float, acf_ratio: float = 0.2,
             max_freq_cm: float = 4000.0, window: str = "hann",
             quantum_correction: str = "harmonic", temperature: float = 300.0,
             smooth_k: int = 10, normalise: bool = True) -> dict:
    n_frames = mu.shape[0]
    L = int(n_frames * acf_ratio)
    L = max(1, min(L, n_frames))
    N2 = 2 * L - 1
    dnu = CM_PER_INV_FS / (N2 * dt_fs)

    acf_full, mu_mean = autocorrelation(mu)
    acf = acf_full[:L]

    w = lag_window(window, L)
    a = acf * w
    a[1:] *= 2.0

    k_max = -1
    for k in range(L):
        if k * dnu <= max_freq_cm:
            k_max = k
        else:
            break
    n_out = k_max + 1
    nu = np.arange(n_out) * dnu

    # The cosine transform, as one matrix product rather than the double loop.
    theta = 2.0 * np.pi * np.outer(nu / dnu, np.arange(L)) / N2
    M = np.cos(theta) @ a

    inten = prefactor(quantum_correction, temperature, nu) * M
    pw = M.copy()
    if smooth_k > 1 and n_out > smooth_k:
        inten = smooth_gaussian(inten, smooth_k)
        pw = smooth_gaussian(pw, smooth_k)

    peak = np.abs(inten).max()
    out = {"nu": nu, "intensity_raw": inten, "power_raw": pw,
           "mu_mean": mu_mean, "n_lag": L, "d_nu": dnu, "peak": peak,
           "resolution": CM_PER_INV_FS / (L * dt_fs)}
    out["intensity"] = inten / peak if (normalise and peak > 0) else inten
    return out


def load_dipoles(path: Path, dt_fallback: float = 0.0) -> tuple[np.ndarray, float]:
    """The dipole series and the interval between frames.

    A run that took its interval from `ir_frame_dt` rather than from `time=`
    tags writes a column of zeros here -- turboGAP records the labels it read,
    and it read none. So a zero interval is not corruption, it is the fallback
    path leaving its trace, and the caller has to supply the same number the
    input file did.
    """
    d = np.loadtxt(path, comments="#")
    t, mu = d[:, 0], d[:, 1:4]
    dt = float((t[-1] - t[0]) / (len(t) - 1))
    if dt <= 0.0:
        if dt_fallback <= 0.0:
            raise ValueError(
                f"{path} carries no time axis (every time_fs is {t[0]:g}), so "
                "the frame interval has to come from the input deck's "
                "ir_frame_dt. Pass it as --dt."
            )
        dt = dt_fallback
    return mu, dt


def check(run: Path) -> int:
    """Recompute a finished run's spectrum from its own dipoles."""
    mu, dt = load_dipoles(run / "ir_fft_dipoles.dat")
    ref = np.loadtxt(run / "ir_fft_spectrum.dat", comments="#")
    got = spectrum(mu, dt)
    n = min(len(ref), len(got["nu"]))
    # The .dat carries ~9 significant digits, so a column whose values run to
    # 4000 cannot agree better than 1e-5 absolute however right the estimator
    # is. The tolerance is per column for that reason, not one number.
    for name, col, tol in (("nu", 0, 1e-8), ("intensity", 1, 1e-9)):
        a, b = ref[:n, col], got[name][:n]
        scale = np.abs(b).max() or 1.0
        err = float(np.abs(a - b).max() / scale)
        print(f"  {name:<10} max |diff| / max|value| = {err:.3e}"
              f"   {'OK' if err < tol else 'MISMATCH'}")
    print(f"  frames {len(mu)}  dt {dt:.4f} fs  lags {got['n_lag']}  "
          f"dnu {got['d_nu']:.4f}  resolution {got['resolution']:.4f} cm^-1")
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "check":
        raise SystemExit(check(Path(sys.argv[2])))
    print(__doc__)
