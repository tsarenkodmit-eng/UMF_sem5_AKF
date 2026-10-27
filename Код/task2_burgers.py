#!/usr/bin/env python3
"""MacCormack solution of inviscid/viscous Burgers Riemann problem.

The requested output is x in [0,4], but the computational domain is extended
to [-6,4] to represent both states of u(x,0)=1-H(x). This avoids placing an
artificial inflow boundary at the initial discontinuity x=0.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc

plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix", "pdf.fonttype": 42})


def flux(u: np.ndarray) -> np.ndarray:
    return 0.5 * u * u


def analytic_inviscid(x: np.ndarray, t: float) -> np.ndarray:
    """Entropy solution: a shock connecting left state 1 to right state 0."""
    if t <= 0.0:
        return np.where(x <= 0.0, 1.0, 0.0)
    return np.where(x <= 0.5 * t, 1.0, 0.0)


def analytic_viscous(x: np.ndarray, t: float, nu: float) -> np.ndarray:
    """Exact Cole-Hopf solution for the Riemann data u_L=1, u_R=0."""
    if nu <= 0.0:
        raise ValueError("nu must be positive for the viscous exact solution")
    if t <= 0.0:
        return np.where(x <= 0.0, 1.0, 0.0)
    q = 2.0 * math.sqrt(nu * t)
    term_right = erfc(-x / q)
    exponent = (t - 2.0 * x) / (4.0 * nu)
    term_left = np.exp(exponent) * erfc((x - t) / q)
    return term_left / (term_left + term_right)


def stability_step(h: float, nu: float, u_max: float, factor: float) -> float:
    """Combined explicit advection and diffusion step bound."""
    u_max = max(float(u_max), 1e-12)
    return factor / (u_max / h + 2.0 * nu / h**2)


def solve_maccormack(nu: float, factor: float, *, h: float = 0.02,
                     t_final: float = 2.0, x_min: float = -6.0,
                     x_max: float = 4.0) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Explicit predictor-corrector MacCormack method with centered viscosity."""
    if nu < 0 or factor not in (0.5, 1.0) or h <= 0 or t_final <= 0:
        raise ValueError("Invalid solver parameters")
    n_intervals = round((x_max - x_min) / h)
    x = np.linspace(x_min, x_max, n_intervals + 1)
    # Keep the lecture's literal nodal convention u_0(x)=1 for x<=0.
    # With trapezoidal nodal integration this puts h/2 extra mass at the jump;
    # the resulting leading O(h) shift is documented separately from scheme error.
    u = np.where(x < 0.0, 1.0, 0.0).astype(float)
    u[np.isclose(x, 0.0, rtol=0.0, atol=1e-12)] = 1.0
    u[0], u[-1] = 1.0, 0.0
    initial_mass = float(np.trapezoid(u, x))

    t = 0.0
    n_steps = 0
    max_courant = 0.0
    max_diffusion_number = 0.0
    max_combined_stability_number = 0.0
    max_abs_u = float(np.max(np.abs(u)))
    min_u = float(np.min(u))
    time_tolerance = 1e-12 * max(1.0, t_final)

    while t_final - t > time_tolerance:
        u_max_now = float(np.max(np.abs(u)))
        remaining = t_final - t
        dt = min(stability_step(h, nu, u_max_now, factor), remaining)
        courant = u_max_now * dt / h
        diffusion_number = nu * dt / h**2
        max_courant = max(max_courant, courant)
        max_diffusion_number = max(max_diffusion_number, diffusion_number)
        max_combined_stability_number = max(
            max_combined_stability_number, courant + 2.0 * diffusion_number
        )
        d = nu * dt / h**2

        # Predictor: forward difference for the conservative flux.
        u_star = u.copy()
        u_star[1:-1] = (
            u[1:-1]
            - (dt / h) * (flux(u[2:]) - flux(u[1:-1]))
            + d * (u[2:] - 2.0 * u[1:-1] + u[:-2])
        )
        u_star[0], u_star[-1] = 1.0, 0.0

        # Corrector: backward flux difference and average of two time levels.
        u_next = u.copy()
        u_next[1:-1] = 0.5 * (
            u[1:-1] + u_star[1:-1]
            - (dt / h) * (flux(u_star[1:-1]) - flux(u_star[:-2]))
            + d * (u_star[2:] - 2.0 * u_star[1:-1] + u_star[:-2])
        )
        u_next[0], u_next[-1] = 1.0, 0.0
        if not np.isfinite(u_next).all():
            raise FloatingPointError(
                f"Non-finite solution at t={t:g}, nu={nu}, factor={factor}"
            )
        u = u_next
        t_next = t + dt
        t = t_final if t_final - t_next <= time_tolerance else t_next
        n_steps += 1
        max_abs_u = max(max_abs_u, float(np.max(np.abs(u))))
        min_u = min(min_u, float(np.min(u)))
        if n_steps > 2_000_000:
            raise RuntimeError("Step limit exceeded; explicit scheme may be unstable")

    diagnostics = {
        "nu": nu,
        "stability_factor": factor,
        "h": h,
        "t_final": t_final,
        "steps": float(n_steps),
        "max_observed_advective_courant": max_courant,
        "max_diffusion_number": max_diffusion_number,
        "max_combined_stability_number": max_combined_stability_number,
        "max_abs_u": max_abs_u,
        "min_u": min_u,
        "initial_value_at_jump": 1.0,
        "initial_trapezoid_mass": initial_mass,
        "initial_mass_excess_over_continuum": initial_mass + x_min,
        "final_trapezoid_mass": float(np.trapezoid(u, x)),
        "conservative_mass_balance_defect": float(np.trapezoid(u, x)) - initial_mass - 0.5 * t_final,
        "inviscid_exact_value_at_shock": 1.0,
    }
    mask = (x >= 0.0) & (x <= 4.0 + 1e-12)
    xr = x[mask]
    exact = analytic_inviscid(xr, t_final) if nu == 0 else analytic_viscous(xr, t_final, nu)
    err = np.abs(u[mask] - exact)
    diagnostics["l1_error_requested_interval"] = float(h * np.sum(err))
    if nu == 0.0:
        shock_node = np.isclose(xr, 0.5 * t_final, rtol=0.0, atol=1e-12)
        linf_errors = err[~shock_node]
        if np.any(shock_node):
            diagnostics["error_at_shock_node_by_left_trace"] = float(np.max(err[shock_node]))
            diagnostics["error_at_shock_node_by_right_trace"] = float(np.max(np.abs(u[mask][shock_node])))
    else:
        linf_errors = err
    diagnostics["linf_error_requested_interval"] = float(np.max(linf_errors))
    return x, u, diagnostics


def write_outputs(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    h, t_final = 0.02, 2.0
    cases = [
        (0.0, "Невязкое уравнение Бюргерса", "ν = 0"),
        (0.02, "Вязкое уравнение Бюргерса", "ν = 0,02"),
        (0.1, "Вязкое уравнение Бюргерса", "ν = 0,1"),
    ]
    diagnostics: list[dict[str, float]] = []
    solutions: dict[tuple[float, float], tuple[np.ndarray, np.ndarray, dict[str, float]]] = {}
    for nu, _title, _label in cases:
        for factor in (1.0, 0.5):
            x, u, diag = solve_maccormack(nu, factor, h=h, t_final=t_final)
            diagnostics.append(diag)
            solutions[(nu, factor)] = (x, u, diag)

    colors = {1.0: "#1c4f85", 0.5: "#ba4b20"}
    # The three panels are embedded at nearly this physical width in the A4
    # report; sizing at placement scale keeps tick labels and legends legible.
    fig, axes = plt.subplots(3, 1, figsize=(6.934, 7.20), constrained_layout=True)
    for ax, (nu, title, subtitle) in zip(axes, cases):
        for factor in (1.0, 0.5):
            x, u, _diag = solutions[(nu, factor)]
            mask = (x >= 0.0) & (x <= 4.0 + 1e-12)
            ax.plot(x[mask], u[mask], color=colors[factor], lw=1.7,
                    ls="-" if factor == 1.0 else "-.",
                    label="Мак-Кормак, κ = " + ("1" if factor == 1.0 else "0,5"))
        xp = np.linspace(0.0, 4.0, 1201)
        ue = analytic_inviscid(xp, t_final) if nu == 0 else analytic_viscous(xp, t_final, nu)
        if nu == 0.0:
            ax.plot([0.0, 1.0, 1.0, 4.0], [1.0, 1.0, 0.0, 0.0],
                    color="#333333", ls="--", lw=1.25, label="Точное решение")
        else:
            ax.plot(xp, ue, color="#333333", ls="--", lw=1.25, label="Точное решение")
        ax.set_xlim(0.0, 4.0)
        ax.set_ylim(-0.18, 1.18)
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("u(x, 2)", fontsize=12)
        ax.set_title(f"{title}, {subtitle}", fontsize=12.5)
        ax.grid(True, alpha=0.22)
        ax.tick_params(axis="both", labelsize=11)
        ax.legend(fontsize=10.5, loc="best")
        if nu == 0.0:
            inset = ax.inset_axes([0.32, 0.30, 0.35, 0.42])
            for factor in (1.0, 0.5):
                x, u, _diag = solutions[(nu, factor)]
                zoom = (x >= 0.72) & (x <= 1.20)
                inset.plot(x[zoom], u[zoom], color=colors[factor], lw=1.3,
                           ls="-" if factor == 1.0 else "-.")
            inset.plot([0.72, 1.0, 1.0, 1.20], [1.0, 1.0, 0.0, 0.0],
                       color="#333333", ls="--", lw=1.0)
            inset.set_xlim(0.72, 1.20)
            inset.set_ylim(-0.12, 1.18)
            inset.set_xticks([0.8, 1.0, 1.2])
            inset.set_yticks([0.0, 1.0])
            inset.tick_params(labelsize=9)
            inset.grid(True, alpha=0.22)
    fig.savefig(output_dir / "task2_burgers_profiles.pdf")
    fig.savefig(output_dir / "task2_burgers_profiles.png", dpi=600)
    fig.savefig(output_dir / "task2_burgers_profiles.svg")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(6.934, 5.222), constrained_layout=True)
    for ax, (nu, _title, subtitle) in zip(axes, cases):
        for factor in (1.0, 0.5):
            x, u, _diag = solutions[(nu, factor)]
            mask = (x >= 0.35) & (x <= 1.65)
            ax.plot(x[mask], u[mask], color=colors[factor], lw=1.7,
                    ls="-" if factor == 1.0 else "-.",
                    label="Мак-Кормак, κ = " + ("1" if factor == 1.0 else "0,5"))
        xp = np.linspace(0.35, 1.65, 1001)
        ue = analytic_inviscid(xp, t_final) if nu == 0 else analytic_viscous(xp, t_final, nu)
        if nu == 0.0:
            ax.plot([0.35, 1.0, 1.0, 1.65], [1.0, 1.0, 0.0, 0.0],
                    color="#333333", ls="--", lw=1.25, label="Точное решение")
        else:
            ax.plot(xp, ue, color="#333333", ls="--", lw=1.25, label="Точное решение")
        ax.axvline(1.0, color="0.45", ls=":", lw=1.0)
        ax.set_xlim(0.35, 1.65)
        ax.set_ylim(-0.18, 1.18)
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("u(x, 2)", fontsize=12)
        ax.set_title(f"Фронт: {subtitle}", fontsize=12.5)
        ax.grid(True, alpha=0.22)
        ax.tick_params(axis="both", labelsize=11)
        ax.legend(fontsize=10.5, loc="best")
    fig.savefig(output_dir / "task2_burgers_shock.pdf")
    fig.savefig(output_dir / "task2_burgers_shock.png", dpi=600)
    fig.savefig(output_dir / "task2_burgers_shock.svg")
    plt.close(fig)

    (output_dir / "task2_diagnostics.json").write_text(
        json.dumps({
            "domain_reported": [0.0, 4.0],
            "internal_domain": [-6.0, 4.0],
            "h": h, "t_final": t_final, "runs": diagnostics
        }, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"runs": diagnostics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("task2_results"))
    args = parser.parse_args()
    result = write_outputs(args.output_dir)
    for row in result["runs"]:
        print(json.dumps(row, ensure_ascii=False))
    print(f"Outputs written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
