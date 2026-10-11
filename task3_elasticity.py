#!/usr/bin/env python3
"""Finite-difference solution of the plane-strain rectangle in Task 3."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix"})


def solve(width: float = 2.0, height: float = 1.0, h: float = 0.025,
          nu: float = 0.3, epsilon: float = 1.0e-5):
    nx, ny = round(width / h), round(height / h)
    if abs(nx * h - width) > 1e-12 or abs(ny * h - height) > 1e-12:
        raise ValueError("The step must divide both side lengths.")
    x = np.linspace(0.0, width, nx + 1)
    y = np.linspace(0.0, height, ny + 1)

    # Plane-strain Lamé ratios; stress is scaled by lambda + 2 mu.
    lam_over_mu = 2.0 * nu / (1.0 - 2.0 * nu)
    A = 1.0 / (lam_over_mu + 2.0)
    C = lam_over_mu / (lam_over_mu + 2.0)
    B = A + C
    ndof = 2 * (nx + 1) * (ny + 1)

    def dof(i: int, j: int, component: int) -> int:
        return 2 * (j * (nx + 1) + i) + component

    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    rhs: list[float] = []

    def equation(terms: dict[int, float], value: float = 0.0) -> None:
        r = len(rhs)
        for col, coefficient in terms.items():
            if coefficient:
                rows.append(r)
                cols.append(col)
                values.append(coefficient)
        rhs.append(value)

    def add(terms: dict[int, float], i: int, j: int, component: int,
            coefficient: float) -> None:
        col = dof(i, j, component)
        terms[col] = terms.get(col, 0.0) + coefficient

    for j in range(ny + 1):
        for component in (0, 1):
            equation({dof(0, j, component): 1.0})

    for j in range(1, ny):
        for i in range(1, nx):
            # Interior Navier equations, multiplied by h_x^2 for scaling.
            ueq: dict[int, float] = {}
            for ii, c in ((i + 1, 1.0), (i, -2.0), (i - 1, 1.0)):
                add(ueq, ii, j, 0, c)
            for jj, c in ((j + 1, A), (j, -2.0 * A), (j - 1, A)):
                add(ueq, i, jj, 0, c)
            cross = B / 4.0
            for ii, jj, c in ((i+1,j+1,1),(i+1,j-1,-1),(i-1,j+1,-1),(i-1,j-1,1)):
                add(ueq, ii, jj, 1, cross * c)
            equation(ueq)

            veq: dict[int, float] = {}
            for jj, c in ((j + 1, 1.0), (j, -2.0), (j - 1, 1.0)):
                add(veq, i, jj, 1, c)
            for ii, c in ((i + 1, A), (i, -2.0 * A), (i - 1, A)):
                add(veq, ii, j, 1, c)
            for ii, jj, c in ((i+1,j+1,1),(i+1,j-1,-1),(i-1,j+1,-1),(i-1,j-1,1)):
                add(veq, ii, jj, 0, cross * c)
            equation(veq)

    # Traction-free lower and upper edges. The normal derivative is second-order
    # one-sided and the tangential derivative is centered.
    for i in range(1, nx):
        for normal_stencil in (((0, -1.5), (1, 2.0), (2, -0.5)),
                               ((ny, 1.5), (ny - 1, -2.0), (ny - 2, 0.5))):
            boundary_j = normal_stencil[0][0]
            txy: dict[int, float] = {}
            for j, coefficient in normal_stencil:
                add(txy, i, j, 0, A * coefficient)
            add(txy, i + 1, boundary_j, 1, A * h / (2.0 * h))
            add(txy, i - 1, boundary_j, 1, -A * h / (2.0 * h))
            equation(txy)

            tyy: dict[int, float] = {}
            for j, coefficient in normal_stencil:
                add(tyy, i, j, 1, coefficient)
            add(tyy, i + 1, boundary_j, 0, C * h / (2.0 * h))
            add(tyy, i - 1, boundary_j, 0, -C * h / (2.0 * h))
            equation(tyy)

    # Applied tractions on x = width; tau(y) = y(1-y).
    for j in range(1, ny):
        normal: dict[int, float] = {}
        for i, coefficient in ((nx, 1.5), (nx - 1, -2.0), (nx - 2, 0.5)):
            add(normal, i, j, 0, coefficient)
        add(normal, nx, j + 1, 1, C * h / (2.0 * h))
        add(normal, nx, j - 1, 1, -C * h / (2.0 * h))
        equation(normal)

        shear: dict[int, float] = {}
        add(shear, nx, j + 1, 0, A * h / (2.0 * h))
        add(shear, nx, j - 1, 0, -A * h / (2.0 * h))
        for i, coefficient in ((nx, 1.5), (nx - 1, -2.0), (nx - 2, 0.5)):
            add(shear, i, j, 1, A * coefficient)
        equation(shear, h * y[j] * (1.0 - y[j]))

    # At each right corner use the two conditions from the loaded edge.
    for normal_stencil in (((0, -1.5), (1, 2.0), (2, -0.5)),
                           ((ny, 1.5), (ny - 1, -2.0), (ny - 2, 0.5))):
        normal: dict[int, float] = {}
        for i, coefficient in ((nx, 1.5), (nx - 1, -2.0), (nx - 2, 0.5)):
            add(normal, i, normal_stencil[0][0], 0, coefficient)
        for j, coefficient in normal_stencil:
            add(normal, nx, j, 1, C * coefficient)
        equation(normal)

        shear: dict[int, float] = {}
        for j, coefficient in normal_stencil:
            add(shear, nx, j, 0, A * coefficient)
        for i, coefficient in ((nx, 1.5), (nx - 1, -2.0), (nx - 2, 0.5)):
            add(shear, i, normal_stencil[0][0], 1, A * coefficient)
        equation(shear, h * y[normal_stencil[0][0]] * (1.0 - y[normal_stencil[0][0]]))

    matrix = coo_matrix((values, (rows, cols)), shape=(ndof, ndof)).tocsr()
    rhs_array = np.asarray(rhs)
    solution = spsolve(matrix, rhs_array)
    residual = matrix @ solution - rhs_array
    residual_inf = float(np.linalg.norm(residual, ord=np.inf))
    if residual_inf >= epsilon:
        raise RuntimeError(f"Scaled linear-system residual {residual_inf:.3e} exceeds epsilon={epsilon:.1e}.")

    U = solution[0::2].reshape(ny + 1, nx + 1)
    V = solution[1::2].reshape(ny + 1, nx + 1)
    return x, y, U, V, {"A": A, "B": B, "C": C, "residual_inf": residual_inf,
                        "nx": nx, "ny": ny, "lambda_over_mu": lam_over_mu}


def make_figure(x: np.ndarray, y: np.ndarray, U: np.ndarray, V: np.ndarray,
                output: Path) -> None:
    X, Y = np.meshgrid(x, y)
    magnitude = np.hypot(U, V)
    scale = 1.0
    fig, ax = plt.subplots(2, 2, figsize=(7.6, 8.2), constrained_layout=True)
    fields = ((U, "Перемещение u", "u"), (V, "Перемещение v", "v"),
              (magnitude, "Модуль перемещения", "|u|"))
    for axis, (field, title, label) in zip(ax.flat, fields):
        contour = axis.contourf(X, Y, field, levels=22, cmap="viridis")
        axis.set(title=title, xlabel="x", ylabel="y", aspect="equal")
        fig.colorbar(contour, ax=axis, label=label, shrink=0.82)

    axis = ax.flat[3]
    step = 4
    for j in range(0, len(y), step):
        axis.plot(X[j, ::step] + scale * U[j, ::step],
                  Y[j, ::step] + scale * V[j, ::step], color="#245b9e", lw=0.85)
    for i in range(0, len(x), step):
        axis.plot(X[::step, i] + scale * U[::step, i],
                  Y[::step, i] + scale * V[::step, i], color="#245b9e", lw=0.85)
    axis.set(title="Деформированная сетка (масштаб решения)", xlabel="x", ylabel="y", aspect="equal")
    fig.savefig(output, dpi=240, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def check_discrete_equations(x: np.ndarray, y: np.ndarray, U: np.ndarray,
                             V: np.ndarray, A: float, B: float,
                             C: float) -> dict[str, float]:
    hx, hy = x[1] - x[0], y[1] - y[0]
    u_xx = (U[1:-1, 2:] - 2.0 * U[1:-1, 1:-1] + U[1:-1, :-2]) / hx**2
    u_yy = (U[2:, 1:-1] - 2.0 * U[1:-1, 1:-1] + U[:-2, 1:-1]) / hy**2
    v_xy = (V[2:, 2:] - V[2:, :-2] - V[:-2, 2:] + V[:-2, :-2]) / (4.0 * hx * hy)
    v_xx = (V[1:-1, 2:] - 2.0 * V[1:-1, 1:-1] + V[1:-1, :-2]) / hx**2
    v_yy = (V[2:, 1:-1] - 2.0 * V[1:-1, 1:-1] + V[:-2, 1:-1]) / hy**2
    u_xy = (U[2:, 2:] - U[2:, :-2] - U[:-2, 2:] + U[:-2, :-2]) / (4.0 * hx * hy)
    pde_x = u_xx + A * u_yy + B * v_xy
    pde_y = v_yy + A * v_xx + B * u_xy

    edge_residuals = []
    for stencil in (((0, -1.5), (1, 2.0), (2, -0.5)),
                    ((len(y)-1, 1.5), (len(y)-2, -2.0), (len(y)-3, 0.5))):
        uy = sum(coefficient * U[j, 1:-1] for j, coefficient in stencil) / hy
        vy = sum(coefficient * V[j, 1:-1] for j, coefficient in stencil) / hy
        j = stencil[0][0]
        vx = (V[j, 2:] - V[j, :-2]) / (2.0 * hx)
        ux = (U[j, 2:] - U[j, :-2]) / (2.0 * hx)
        edge_residuals.extend((A * (uy + vx), vy + C * ux))

    ux_right = (3.0 * U[:, -1] - 4.0 * U[:, -2] + U[:, -3]) / (2.0 * hx)
    vx_right = (3.0 * V[:, -1] - 4.0 * V[:, -2] + V[:, -3]) / (2.0 * hx)
    uy_right = np.empty_like(y)
    vy_right = np.empty_like(y)
    uy_right[1:-1] = (U[2:, -1] - U[:-2, -1]) / (2.0 * hy)
    vy_right[1:-1] = (V[2:, -1] - V[:-2, -1]) / (2.0 * hy)
    uy_right[0] = (-3.0 * U[0, -1] + 4.0 * U[1, -1] - U[2, -1]) / (2.0 * hy)
    uy_right[-1] = (3.0 * U[-1, -1] - 4.0 * U[-2, -1] + U[-3, -1]) / (2.0 * hy)
    vy_right[0] = (-3.0 * V[0, -1] + 4.0 * V[1, -1] - V[2, -1]) / (2.0 * hy)
    vy_right[-1] = (3.0 * V[-1, -1] - 4.0 * V[-2, -1] + V[-3, -1]) / (2.0 * hy)
    applied_shear = y * (1.0 - y)
    right_xx = ux_right + C * vy_right
    right_xy = A * (uy_right + vx_right) - applied_shear

    left_uy = np.gradient(U[:, 0], hy, edge_order=2)
    left_vx = (-3.0 * V[:, 0] + 4.0 * V[:, 1] - V[:, 2]) / (2.0 * hx)
    left_reaction = -float(np.trapezoid(A * (left_uy + left_vx), y))
    return {
        "max_interior_equilibrium_residual": float(max(np.max(np.abs(pde_x)), np.max(np.abs(pde_y)))),
        "max_free_edge_traction_residual": float(max(np.max(np.abs(a)) for a in edge_residuals)),
        "max_loaded_edge_traction_residual": float(max(np.max(np.abs(right_xx)), np.max(np.abs(right_xy)))),
        "postprocessed_clamp_shear_reaction": left_reaction,
        "estimated_shear_resultant_imbalance": left_reaction + 1.0 / 6.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("task3_results"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    x, y, U, V, info = solve()
    X, Y = np.meshgrid(x, y)
    magnitude = np.hypot(U, V)
    imax, jmax = np.unravel_index(np.argmax(magnitude), magnitude.shape)
    checks = check_discrete_equations(x, y, U, V, info["A"], info["B"], info["C"])
    diagnostics = {
        "domain": [0.0, 2.0, 0.0, 1.0], "mesh_spacing": [0.025, 0.025],
        "poisson_ratio": 0.3, "plane_model": "plane strain per lecture Navier system",
        "stress_scale": "lambda + 2 mu", "traction": "sigma_xy_tilde = y(1-y), sigma_xx_tilde = 0 at x=2",
        "coefficients": {key: info[key] for key in ("A", "B", "C", "lambda_over_mu")},
        "nodes": int(len(x) * len(y)), "degrees_of_freedom": int(2 * len(x) * len(y)),
        "scaled_system_residual_inf": info["residual_inf"], "tolerance": 1e-5,
        "max_abs_u": float(np.max(np.abs(U))), "max_abs_v": float(np.max(np.abs(V))),
        "max_displacement": float(magnitude[imax, jmax]),
        "max_displacement_location": [float(X[imax, jmax]), float(Y[imax, jmax])],
        "resultant_shear_load": 1.0 / 6.0,
        "reaction_estimate_method": "Trapezoidal integral of A*(u_y+v_x) recovered by finite differences on x=0; mesh-sensitive near the fixed/free corner.",
        "discrete_checks": checks,
    }
    with (args.output_dir / "task3_diagnostics.json").open("w", encoding="utf-8") as stream:
        json.dump(diagnostics, stream, ensure_ascii=False, indent=2)
    with (args.output_dir / "task3_solution.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["x", "y", "u", "v", "magnitude"])
        for j, yy in enumerate(y):
            for i, xx in enumerate(x):
                writer.writerow([f"{xx:.6f}", f"{yy:.6f}", f"{U[j,i]:.12g}",
                                 f"{V[j,i]:.12g}", f"{magnitude[j,i]:.12g}"])
    make_figure(x, y, U, V, args.output_dir / "task3_elasticity_fields.png")
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
