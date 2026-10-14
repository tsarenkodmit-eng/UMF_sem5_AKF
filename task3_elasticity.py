#!/usr/bin/env python3
"""Conservative finite differences for the lecture's plane-strain Task 3.

The unknowns U,V are nodal displacements on the assigned rectangular mesh.
Interior equations are the centered Navier differences from the lecture.
Boundary equations follow the discrete virtual-work balance, including both
faces at each corner. Forces/moments and recovered stresses are checked
separately: an algebraic residual is not a pointwise stress accuracy estimate.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import bmat, diags, kron
from scipy.sparse.linalg import spsolve

plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix"})


def difference_operators(n: int, h: float):
    """Trapezoidal H, SBP first difference D, narrow Neumann stiffness L."""
    weights = np.full(n + 1, h)
    weights[[0, -1]] = h / 2.0
    H = diags(weights, format="csr")
    D = diags([-np.ones(n), np.ones(n)], [-1, 1],
              shape=(n + 1, n + 1), format="lil") / (2.0 * h)
    D[0, 0], D[0, 1] = -1.0 / h, 1.0 / h
    D[n, n - 1], D[n, n] = -1.0 / h, 1.0 / h
    L = diags([-np.ones(n), np.r_[1.0, np.full(n - 1, 2.0), 1.0],
               -np.ones(n)], [-1, 0, 1], format="csr") / h
    return H, D.tocsr(), L


def assemble(h: float = 0.025, nu: float = 0.3):
    nx, ny = round(2.0 / h), round(1.0 / h)
    if nx < 2 or ny < 2 or abs(nx * h - 2.0) > 1e-12 or abs(ny * h - 1.0) > 1e-12:
        raise ValueError("h must divide both side lengths and give at least two cells.")
    if not (-1.0 < nu < 0.5):
        raise ValueError("Plane-strain Poisson ratio must lie in (-1,0.5).")
    x, y = np.linspace(0.0, 2.0, nx + 1), np.linspace(0.0, 1.0, ny + 1)
    A = (1.0 - 2.0 * nu) / (2.0 * (1.0 - nu))
    C = nu / (1.0 - nu)
    B = A + C
    Hx, Dx, Lx = difference_operators(nx, h)
    Hy, Dy, Ly = difference_operators(ny, h)
    Ix, Iy = diags(np.ones(nx + 1)), diags(np.ones(ny + 1))
    H = kron(Hy, Hx, format="csr")
    D_X, D_Y = kron(Iy, Dx, format="csr"), kron(Dy, Ix, format="csr")
    K_uu = kron(Hy, Lx) + A * kron(Ly, Hx)
    K_vv = A * kron(Hy, Lx) + kron(Ly, Hx)
    K_uv = C * (D_X.T @ H @ D_Y) + A * (D_Y.T @ H @ D_X)
    K = bmat([[K_uu, K_uv], [K_uv.T, K_vv]], format="csr")

    # Integral of q(y)=y(1-y) times the linear boundary nodal hats.
    # These loads include both right endpoints and sum to exactly 1/6.
    boundary_load = h * y * (1.0 - y) - h**3 / 6.0
    boundary_load[[0, -1]] = h**2 * (2.0 - h) / 12.0
    nodes = (nx + 1) * (ny + 1)
    f = np.zeros(2 * nodes)
    right = nx + np.arange(ny + 1) * (nx + 1)
    f[nodes + right] = boundary_load
    left = np.arange(ny + 1) * (nx + 1)
    fixed = np.r_[left, nodes + left]
    free = np.setdiff1d(np.arange(2 * nodes), fixed)
    return x, y, K, f, fixed, free, {"A": A, "B": B, "C": C, "H": H,
                                    "nx": nx, "ny": ny, "left": left}


def recovered_stresses(U, V, h, A, C):
    """Independent nodal recovery, second-order one-sided on the boundary."""
    uy, ux = np.gradient(U, h, h, edge_order=2)
    vy, vx = np.gradient(V, h, h, edge_order=2)
    return ux + C * vy, vy + C * ux, A * (uy + vx)


def solve(h: float = 0.025, nu: float = 0.3, epsilon: float = 1e-5):
    x, y, K, f, fixed, free, op = assemble(h, nu)
    z = np.zeros(len(f))
    z[free] = spsolve(K[free, :][:, free], f[free])
    residual = K @ z - f
    n = len(f) // 2
    U, V = z[:n].reshape(len(y), len(x)), z[n:].reshape(len(y), len(x))
    Hdiag = np.tile(op["H"].diagonal(), 2)
    force_residual = float(np.max(np.abs(residual[free])))
    density_residual = float(np.max(np.abs(residual[free] / Hdiag[free])))
    if not np.isfinite(z).all() or density_residual > epsilon:
        raise RuntimeError(f"Nodal equilibrium residual {density_residual:g} exceeds {epsilon:g}.")
    left = op["left"]
    rx, ry = residual[left], residual[n + left]
    force_x, force_y = float(rx.sum()), float(ry.sum())
    moment = float(-np.dot(y, rx))
    magnitude = np.hypot(U, V)
    jmax, imax = np.unravel_index(np.argmax(magnitude), magnitude.shape)
    sigxx, sigyy, sigxy = recovered_stresses(U, V, h, op["A"], op["C"])
    independent_ry = -float(np.trapezoid(sigxy[:, 0], y))

    # Navier equations checked independently at all interior nodes.
    uxx = (U[1:-1, 2:] - 2*U[1:-1, 1:-1] + U[1:-1, :-2]) / h**2
    uyy = (U[2:, 1:-1] - 2*U[1:-1, 1:-1] + U[:-2, 1:-1]) / h**2
    vxx = (V[1:-1, 2:] - 2*V[1:-1, 1:-1] + V[1:-1, :-2]) / h**2
    vyy = (V[2:, 1:-1] - 2*V[1:-1, 1:-1] + V[:-2, 1:-1]) / h**2
    vxy = (V[2:, 2:] - V[2:, :-2] - V[:-2, 2:] + V[:-2, :-2]) / (4*h**2)
    uxy = (U[2:, 2:] - U[2:, :-2] - U[:-2, 2:] + U[:-2, :-2]) / (4*h**2)
    pde = max(np.abs(uxx + op["A"]*uyy + op["B"]*vxy).max(),
              np.abs(vyy + op["A"]*vxx + op["B"]*uxy).max())
    X, Y = np.meshgrid(x, y)
    rotation = np.r_[-Y.ravel(), X.ravel()]
    rigid = np.max(np.abs(K @ rotation))
    checks = {
        "mesh_spacing": h, "nodes": n, "degrees_of_freedom": 2*n,
        "A": op["A"], "B": op["B"], "C": op["C"],
        "force_residual_inf_all_free_nodes_including_right_corners": force_residual,
        "equilibrium_residual_per_nodal_area_inf": density_residual,
        "interior_navier_residual_inf": float(pde), "tolerance": epsilon,
        "applied_force_y": float(f[n:].sum()), "applied_moment_about_origin": float(2*f[n:].sum()),
        "clamp_reaction_x": force_x, "clamp_reaction_y": force_y,
        "clamp_moment_about_origin": moment,
        "force_balance_error_x": force_x,
        "force_balance_error_y": force_y + float(f[n:].sum()),
        "moment_balance_error": moment + float(2*f[n:].sum()),
        "rigid_rotation_stiffness_residual": float(rigid),
        "max_abs_u": float(np.abs(U).max()), "max_abs_v": float(np.abs(V).max()),
        "max_displacement": float(magnitude[jmax, imax]),
        "max_displacement_location": [float(x[imax]), float(y[jmax])],
        "recovered_clamp_reaction_y": independent_ry,
        "recovered_clamp_reaction_relative_error": abs(independent_ry + 1/6)/(1/6),
        "recovered_sigma_yy_at_right_corners": sigyy[[0, -1], -1].tolist(),
        "recovered_sigma_yy_max_on_free_edges_including_right_corners": float(np.abs(sigyy[[0, -1], 1:]).max()),
        "recovered_sigma_xy_max_on_free_edges_including_right_corners": float(np.abs(sigxy[[0, -1], 1:]).max()),
        "recovered_sigma_yy_at_fixed_free_corners": sigyy[[0, -1], 0].tolist(),
        "stress_recovery_note": "Boundary loads enter integral nodal balances; recovered derivatives are approximate pointwise stresses, especially near fixed/free corners. Their errors are not algebraic residuals.",
    }
    return x, y, U, V, rx, ry, checks


def make_figures(x, y, U, V, output_dir):
    # Conventional 2D line profiles; no filled contours, heatmaps or colorbars.
    fig, ax = plt.subplots(figsize=(6.934, 2.20), constrained_layout=True)
    ax.plot([0, 2, 2, 0, 0], [0, 0, 1, 1, 0], color='black', lw=1.5)
    for yy in np.linspace(0, 1, 11):
        ax.plot([-.07, 0], [yy-.045, yy], color='black', lw=.8)
    for yy in np.linspace(.10, .80, 6):
        ax.annotate('', xy=(2.04, yy + .35*yy*(1-yy)), xytext=(2.04, yy),
                    arrowprops={'arrowstyle': '->', 'color': '#1c4f85', 'lw': 1.3})
    ax.text(-.14, .50, 'u=v=0', ha='right', va='center', fontsize=11)
    ax.text(1, 1.07, 'Свободный край', ha='center', fontsize=11)
    ax.text(1, -.20, 'Свободный край', ha='center', fontsize=11)
    ax.text(2.16, .52, 'q(y)=y(1-y)', fontsize=11, va='center')
    ax.set(xlim=(-.45, 2.70), ylim=(-.28, 1.24), aspect='equal')
    ax.axis('off')
    fig.savefig(output_dir / 'task3_geometry.png', dpi=240)
    fig.savefig(output_dir / 'task3_geometry.svg')
    plt.close(fig)
    slices = (0.0, 0.25, 0.5)
    styles = (("#1c4f85", "-"), ("#ba4b20", "--"), ("#333333", ":"))
    for field, component in ((U, "u"), (V, "v")):
        fig, ax = plt.subplots(figsize=(6.934, 3.20), constrained_layout=True)
        for ys, (color, style) in zip(slices, styles):
            j = int(np.argmin(np.abs(y - ys)))
            ax.plot(x, field[j], color=color, ls=style, lw=1.9, label=f"y={ys:g}".replace(".", ","))
        ax.set(xlim=(0, 2), xlabel="x", ylabel=component,
               title=f"{component}(x, y) на горизонтальных сечениях")
        ax.grid(True, alpha=0.22)
        ax.title.set_fontsize(12.5)
        ax.legend(fontsize=10.5)
        ax.tick_params(labelsize=11)
        ax.xaxis.label.set_size(12)
        ax.yaxis.label.set_size(12)
        fig.savefig(output_dir / f"task3_{component}_profiles.png", dpi=240)
        fig.savefig(output_dir / f"task3_{component}_profiles.svg")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("task3_results"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for h in (0.025, 0.0125, 0.00625):
        result = solve(h)
        runs.append(result[-1])
        if h == 0.025:
            x, y, U, V, rx, ry, _ = result
            with (args.output_dir / "task3_solution.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["x", "y", "U", "V"])
                for j, yy in enumerate(y):
                    for i, xx in enumerate(x):
                        writer.writerow([xx, yy, U[j, i], V[j, i]])
            with (args.output_dir / "task3_clamp_reactions.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["y", "nodal_Rx", "nodal_Ry"])
                writer.writerows(zip(y, rx, ry))
            make_figures(x, y, U, V, args.output_dir)
    summary = {"model": "plane strain, scaled stress sigma/(lambda+2mu)",
               "method": "conservative finite differences with integral traction closure",
               "load": "q(y)=y(1-y) on x=2", "runs": runs}
    (args.output_dir / "task3_diagnostics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
