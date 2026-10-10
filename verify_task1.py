#!/usr/bin/env python3
"""Regression and convergence checks for Task 1.

Run: python verify_task1.py
The refinement sequence is time-aligned (T/tau is an integer), so the
target profile does not require temporal interpolation for this check.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from task1_wave import Parameters, analytic_displacement, material_constants, solve_finite_differences


def relative_boundary_residual(p: Parameters) -> float:
    constants = material_constants(p)
    dr = 1e-5 * p.cavity_radius
    radii = np.array([p.cavity_radius - dr, p.cavity_radius, p.cavity_radius + dr])
    u = analytic_displacement(radii, p.final_time, p, constants)
    ur = (u[2] - u[0]) / (2.0 * dr)
    sigma_rr = constants["K"] * ur + 2.0 * constants["lambda"] * u[1] / p.cavity_radius
    return float(abs(sigma_rr + p.pressure) / p.pressure)


def main() -> None:
    p0 = Parameters()
    constants = material_constants(p0)
    assert math.isclose(constants["nu"], 0.35, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(constants["lambda"], 4.666666666666667e9, rel_tol=1e-12)
    assert math.isclose(constants["cp"], 2194.26862868, rel_tol=1e-10)

    radii = np.array([p0.cavity_radius, p0.cavity_radius + 1.0,
                      p0.cavity_radius + constants["cp"] * p0.final_time + 1.0])
    assert np.max(np.abs(analytic_displacement(radii, 0.0, p0, constants))) == 0.0
    bc_residual = relative_boundary_residual(p0)
    assert bc_residual < 1e-8

    # Actual assignment grid: h/R = 0.01 and tau = h/C_p.
    r, u_fd, diagnostics = solve_finite_differences(p0)
    u_exact = analytic_displacement(r, p0.final_time, p0, constants)
    inside = r <= diagnostics["wavefront_radius"] + 1e-12
    error = np.abs(u_fd[inside] - u_exact[inside])
    base = {
        "h_m": diagnostics["h"],
        "tau_s": diagnostics["dt"],
        "courant": diagnostics["courant"],
        "max_error_m": float(np.max(error)),
        "relative_error": float(np.max(error) / np.max(np.abs(u_exact[inside]))),
        "outer_radius_m": diagnostics["outer_radius"],
        "wavefront_radius_m": diagnostics["wavefront_radius"],
        "analytic_boundary_relative_residual": bc_residual,
    }
    assert base["courant"] == 1.0
    assert base["max_error_m"] < 1e-5

    # Independent convergence sequence with T/tau = 2200, 4400, 8800.
    refinements = []
    for nt in (2200, 4400, 8800):
        pref = Parameters(h_over_R=constants["cp"] * p0.final_time /
                          (nt * p0.cavity_radius))
        rr, numerical, diag = solve_finite_differences(pref)
        exact = analytic_displacement(rr, pref.final_time, pref, material_constants(pref))
        mask = rr <= diag["wavefront_radius"] + 1e-12
        e = np.abs(numerical[mask] - exact[mask])
        refinements.append({
            "time_steps": nt,
            "h_m": diag["h"],
            "max_error_m": float(np.max(e)),
            "rms_error_m": float(np.sqrt(np.mean(e**2))),
        })
    for i in range(len(refinements) - 1):
        refinements[i]["observed_order_max"] = math.log(
            refinements[i]["max_error_m"] / refinements[i + 1]["max_error_m"], 2.0
        )
        refinements[i]["observed_order_rms"] = math.log(
            refinements[i]["rms_error_m"] / refinements[i + 1]["rms_error_m"], 2.0
        )
        assert refinements[i + 1]["max_error_m"] < refinements[i]["max_error_m"]
        assert refinements[i + 1]["rms_error_m"] < refinements[i]["rms_error_m"]
        assert 1.8 < refinements[i]["observed_order_max"] < 2.2
        assert 1.8 < refinements[i]["observed_order_rms"] < 2.2

    result = {"status": "PASS", "base_mesh": base, "time_aligned_refinement": refinements}
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    (output_dir / "task1_verification.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    print("All Task 1 checks passed.")


if __name__ == "__main__":
    main()
