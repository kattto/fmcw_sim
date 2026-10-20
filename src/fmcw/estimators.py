# Implements the estimators of interest. Includes single-tone and multi-tone

import scipy
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def fft_peak(x):
    return np.argmax(np.abs(np.fft.fft(x)))


def aboutanios_mulgrew(x, n_iter=2):
    N = len(x)
    X = np.fft.fft(x)
    n_axis = np.arange(N)

    peak_bin = np.argmax(np.abs(X))

    delta = 0.0
    for _ in range(n_iter):
        p_plus = (peak_bin + delta) + 1/2
        p_minus = (peak_bin + delta) - 1/2

        exp_plus = np.exp(-1j * 2 * np.pi * p_plus * n_axis / N)
        exp_minus = np.exp(-1j * 2 * np.pi * p_minus * n_axis / N)

        val_plus = np.dot(x, exp_plus)
        val_minus = np.dot(x, exp_minus)

        delta += 1/2 * np.real((val_plus + val_minus) / (val_plus - val_minus))

    return peak_bin + delta


def fiib(x, n_iter=2, L=1):
    N = len(x)
    X = np.fft.fft(x)

    n_axis = np.arange(N)

    nu_hat = np.zeros(L)
    alpha_hat = np.zeros(L, dtype=np.complex128)

    for q in range(n_iter):
        for l in range(L):

            if q == 0:
                X_l = X.copy()
                # Step 1: Subtract other targets
                for t in range(L):
                    if t != l:
                        target_tone = alpha_hat[t] * np.exp(1j * 2.0 * np.pi * nu_hat[t] * n_axis / N)
                        target_spectrum = np.fft.fft(target_tone)
                        X_l -= target_spectrum

                # Step 2: Obtain rough peak
                nu_hat[l] = np.argmax(np.abs(X_l))

            # Step 3-4: Interpolation via A&M technique
            p_plus = nu_hat[l] + 1/2
            exp_plus = np.exp(-1j * 2.0 * np.pi * p_plus * n_axis / N)
            X_plus = np.dot(x, exp_plus)

            p_minus = nu_hat[l] - 1/2
            exp_minus = np.exp(-1j * 2.0 * np.pi * p_minus * n_axis / N)
            X_minus = np.dot(x, exp_minus)

            for i in range(L):
                if i != l:
                    unit_tone = np.exp(1j * 2.0 * np.pi * nu_hat[i] * n_axis / N)

                    S_plus = np.dot(unit_tone, exp_plus)
                    X_plus -= alpha_hat[i] * S_plus

                    S_minus = np.dot(unit_tone, exp_minus)
                    X_minus -= alpha_hat[i] * S_minus

            # Step 5-6: Sub-bin offset + Update frequency estimate
            h = 1/2 * np.real( (X_plus + X_minus) / (X_plus - X_minus) )
            nu_hat[l] += h
            
            # Step 7: Update amplitude estimate
            X_nu_l = np.dot(x, np.exp(-1j * 2.0 * np.pi * nu_hat[l] * n_axis / N))
            for i in range(L):
                if i != l:
                    unit_tone = np.exp(1j * 2.0 * np.pi * nu_hat[i] * n_axis / N)
                    exp = np.exp(-1j * 2.0 * np.pi * nu_hat[l] * n_axis / N)
                    S_i = np.dot(unit_tone, exp)
                    X_nu_l -= alpha_hat[i] * S_i
            
            alpha_hat[l] = 1/N * X_nu_l

    # Needs to differentiate between single target and target separation use cases
    if L == 1:
        return nu_hat[0]
    else:
        return nu_hat



def root_music(x, L=1):
    N = len(x)
    P = N//2 - 1                # FIIB paper uses N/2 - 1
    S = N - P + 1
    nu_hat = np.zeros(L)

    # Create covariance matrix
    Y = np.ascontiguousarray(sliding_window_view(x, window_shape=P))      # Returns (S, P)
    R_hat = (Y.T @ Y.conj()) / S                    # (P, S) @ (S, P)

    # # Decompose the covariance matrix
    # eigvals, eigvecs = np.linalg.eigh(R_hat)

    # # Extract the noise subspace (first P-L columns)
    # E_n = eigvecs[:, :P-L]

    # # Projection matrix
    # C = E_n @ E_n.conj().T
    eigvals, E_s = scipy.linalg.eigh(R_hat, subset_by_index=[P - L, P - 1])
    C = np.eye(P) - E_s @ E_s.conj().T

    # Sum diagonals (top right to bottom left)
    coeffs = np.zeros(int(2*P - 1), dtype=np.complex128)
    for i, diag in enumerate(range(P-1, -P, -1)):
        coeffs[i] = np.trace(C, offset=diag)

    # Find roots
    roots_cplx = np.roots(coeffs)
    roots = []
    for root in roots_cplx:
        if np.abs(root) < 1:
            roots.append(root)
    roots_tgts = sorted(roots, key=abs, reverse=True)[:L]

    for i, tgt in enumerate(roots_tgts):
        tgt_bin = np.angle(tgt) / (2.0 * np.pi) * N
        nu_hat[i] = tgt_bin if tgt_bin >= 0 else tgt_bin + N
    nu_hat.sort()

    # Needs to differentiate between single target and target separation use cases
    if L == 1:
        return nu_hat[0]
    else:
        return nu_hat



def esprit(x, L=1):
    N = len(x)
    P = N//2 - 1                # FIIB paper uses N/2 - 1
    S = N - P + 1
    nu_hat = np.zeros(L)

    # Create covariance matrix
    Y = np.ascontiguousarray(sliding_window_view(x, window_shape=P))      # Returns (S, P)
    R_hat = (Y.T @ Y.conj()) / S                    # (P, S) @ (S, P)

    # # Decompose the covariance matrix
    # eigvals, eigvecs = np.linalg.eigh(R_hat)

    # # Extract the signal subspace (last L columns) size (P, L)
    # E_s = eigvecs[:, -L:]
    eigvals, E_s = scipy.linalg.eigh(R_hat, subset_by_index=[P - L, P - 1])

    # Shifted copies
    E_1 = E_s[:-1, :]          # Without last row
    E_2 = E_s[1:, :]           # Without first row

    # Solve rotation: L x L matrix psi
    psi = np.linalg.lstsq(E_1, E_2)[0]

    # Find eigenvalues of psi
    tgts = np.linalg.eigvals(psi)

    # Convert to bins
    for i, tgt in enumerate(tgts):
        tgt_bin = np.angle(tgt) / (2.0 * np.pi) * N
        nu_hat[i] = tgt_bin if tgt_bin >= 0 else tgt_bin + N
    nu_hat.sort()

    # Needs to differentiate between single target and target separation use cases
    if L == 1:
        return nu_hat[0]
    else:
        return nu_hat