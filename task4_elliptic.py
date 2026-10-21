#!/usr/bin/env python3
"""Canonical reduction, iterative finite differences and Green identities."""
from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.sparse import csr_matrix, diags, tril, triu
from scipy.sparse.linalg import spsolve_triangular
from scipy.special import k0, k1, y0, y1

plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix", "pdf.fonttype": 42})

CASES = (1, -1, 0)
PRINTED_A = (
    ("1.547619047619047", "-.114285714285714", ".904761904761905", "-.228571428571429", ".738095238095238"),
    ("-.114285714285714", ".114285714285714", ".228571428571429", ".228571428571429", "-.571428571428571"),
    (".904761904761905", ".228571428571429", ".190476190476190", ".457142857142857", "-1.476190476190476"),
    ("-.228571428571429", ".228571428571429", ".457142857142857", ".457142857142857", "-1.142857142857143"),
    (".738095238095238", "-.571428571428571", "-1.476190476190476", "-1.142857142857143", "2.690476190476191"),
)
PRINTED_B = ("1.470526299866703", "-1.881907599231525", "-.704984621975347", ".708320756536520", "2.880495543856732")


def canonical_reduction():
    mp.mp.dps = 75
    a = mp.matrix(PRINTED_A)
    b = mp.matrix(PRINTED_B)
    eig, qd = mp.eigsy(a)
    for j in range(5):
        if qd[0,j] < 0:
            for i in range(5):
                qd[i,j] = -qd[i,j]
    sd = mp.diag([1/mp.sqrt(abs(eig[i])) for i in range(5)])
    ad = sd*qd.T
    target = mp.diag([mp.sign(eig[i]) for i in range(5)])
    norm_inf = lambda matrix: max(sum(abs(matrix[i,j]) for j in range(matrix.cols)) for i in range(matrix.rows))
    strings = lambda matrix: [[mp.nstr(matrix[i,j],35) for j in range(matrix.cols)] for i in range(matrix.rows)]
    return {
        "printed_A": PRINTED_A, "printed_b": PRINTED_B,
        "literal_A_eigenvalues": [mp.nstr(v,35) for v in eig],
        "literal_A_inertia_positive_negative_zero": [3,2,0],
        "literal_Q": strings(qd), "literal_alpha": strings(ad), "literal_gamma": strings(ad*b),
        "orthogonal_first_order": strings(qd.T*b),
        "normalizing_scales": [mp.nstr(mp.sqrt(abs(v)),35) for v in eig],
        "orthogonal_residual_inf": mp.nstr(norm_inf(qd.T*a*qd-mp.diag(eig)),12),
        "orthogonality_residual_inf": mp.nstr(norm_inf(qd.T*qd-mp.eye(5)),12),
        "literal_canonical_residual_inf": mp.nstr(norm_inf(ad*a*ad.T-target),12),
    }


GREEN_POINTS = ((1.1, 0.3), (1.1, -0.3), (1.1, 1.0))


def source_term(x, y, c):
    return 6.0 * y * (x + 1.0) + c * y * (x**3 + y**2)


def exact_solution(x, y):
    return y * (x**3 + y**2)


def green_value(radius, c):
    radius = np.asarray(radius)
    if c == 0:
        return -np.log(radius) / (2.0 * np.pi)
    if c == 1:
        return -y0(radius) / 4.0
    if c == -1:
        return k0(radius) / (2.0 * np.pi)
    raise ValueError("c must be -1, 0 or 1")


def green_radial_derivative(radius, c):
    radius = np.asarray(radius)
    if c == 0:
        return -1.0 / (2.0 * np.pi * radius)
    if c == 1:
        return y1(radius) / 4.0
    if c == -1:
        return -k1(radius) / (2.0 * np.pi)
    raise ValueError("c must be -1, 0 or 1")


def normal_kernel(points, observation_point, normal, c):
    difference = np.asarray(points) - np.asarray(observation_point)
    radius = np.linalg.norm(difference, axis=-1)
    return green_radial_derivative(radius, c) * (difference @ normal) / radius


@lru_cache(maxsize=None)
def _gauss(order):
    abscissa, weight = leggauss(order)
    return (abscissa + 1.0) / 2.0, weight / 2.0


def _validate_grid(x, y, u):
    x, y, u = np.asarray(x), np.asarray(y), np.asarray(u)
    if u.shape != (len(y), len(x)) or min(len(x), len(y)) < 4:
        raise ValueError("U must have shape (len(y), len(x)); at least four nodes are required")
    if not np.allclose([x[0], x[-1], y[0], y[-1]], [0.0, 2.0, 0.0, 1.0]):
        raise ValueError("The assignment domain must be [0,2] x [0,1]")
    if not np.allclose(np.diff(x), x[1]-x[0]) or not np.allclose(np.diff(y), y[1]-y[0]):
        raise ValueError("Uniform grids are required by the recovery formulas")
    return x, y, u


def build_boundary_traces(x, y, u, exact_reference=False):
    x, y, u = _validate_grid(x, y, u)
    hy = y[1] - y[0]
    if exact_reference:
        bottom_q = -x**3
        top_q = x**3 + 3.0
        left_u = y**3
        right_u = y * (8.0 + y**2)
    else:
        bottom_q = -(-3.0*u[0, :] + 4.0*u[1, :] - u[2, :]) / (2.0*hy)
        top_q = (3.0*u[-1, :] - 4.0*u[-2, :] + u[-3, :]) / (2.0*hy)
        left_u = u[:, 0]
        right_u = u[:, -1]
    definitions = (
        ("bottom", x, np.zeros_like(x), bottom_q, (0., 0.), (1., 0.), (0., -1.)),
        ("right", y, right_u, 12.0*y, (2., 0.), (0., 1.), (1., 0.)),
        ("top", x, x**3+1.0, top_q, (0., 1.), (1., 0.), (0., 1.)),
        ("left", y, left_u, np.zeros_like(y), (0., 0.), (0., 1.), (-1., 0.)),
    )
    sides = []
    for name, nodes, values_u, values_q, origin, tangent, normal in definitions:
        sides.append({
            "name": name,
            "nodes": nodes,
            "u": lambda t, nodes=nodes, values=values_u: np.interp(t, nodes, values),
            "q": lambda t, nodes=nodes, values=values_q: np.interp(t, nodes, values),
            "values_u": np.asarray(values_u),
            "values_q": np.asarray(values_q),
            "origin": np.asarray(origin),
            "tangent": np.asarray(tangent),
            "normal": np.asarray(normal),
        })
    # Preserve the prescribed continuous data. Only the missing traces are
    # reconstructed linearly from the numerical nodes.
    sides[0]["u"] = lambda t: np.zeros_like(np.asarray(t), dtype=float)
    sides[2]["u"] = lambda t: np.asarray(t)**3 + 1.0
    sides[1]["q"] = lambda t: 12.0*np.asarray(t)
    sides[3]["q"] = lambda t: np.zeros_like(np.asarray(t), dtype=float)
    if exact_reference:
        sides[0]["q"] = lambda t: -np.asarray(t)**3
        sides[2]["q"] = lambda t: np.asarray(t)**3 + 3.0
        sides[1]["u"] = lambda t: np.asarray(t)*(8.0+np.asarray(t)**2)
        sides[3]["u"] = lambda t: np.asarray(t)**3
    return sides


def boundary_recovery_diagnostics(x, y, u):
    x, y, u = _validate_grid(x, y, u)
    hx, hy = x[1]-x[0], y[1]-y[0]
    left_derivative = (-3.0*u[:, 0]+4.0*u[:, 1]-u[:, 2])/(2.0*hx)
    right_derivative = (3.0*u[:, -1]-4.0*u[:, -2]+u[:, -3])/(2.0*hx)
    bottom_q = -(-3.0*u[0, :]+4.0*u[1, :]-u[2, :])/(2.0*hy)
    top_q = (3.0*u[-1, :]-4.0*u[-2, :]+u[-3, :])/(2.0*hy)
    return {
        "bottom_dirichlet_defect_max": float(np.max(np.abs(u[0, :]))),
        "top_dirichlet_defect_max": float(np.max(np.abs(u[-1, :]-(x**3+1.0)))),
        "left_neumann_defect_excluding_corners": float(np.max(np.abs(left_derivative[1:-1]))),
        "right_neumann_defect_excluding_corners": float(np.max(np.abs(right_derivative[1:-1]-12.0*y[1:-1]))),
        "left_neumann_corner_defects": left_derivative[[0, -1]].tolist(),
        "right_neumann_corner_defects": (right_derivative-12.0*y)[[0, -1]].tolist(),
        "bottom_recovered_q_endpoints": bottom_q[[0, -1]].tolist(),
        "top_recovered_q_endpoints": top_q[[0, -1]].tolist(),
        "bottom_q_error_against_reference_max": float(np.max(np.abs(bottom_q+x**3))),
        "top_q_error_against_reference_max": float(np.max(np.abs(top_q-(x**3+3.0)))),
        "left_u_error_against_reference_max": float(np.max(np.abs(u[:, 0]-y**3))),
        "right_u_error_against_reference_max": float(np.max(np.abs(u[:, -1]-y*(8.0+y**2)))),
        "corner_policy": "Horizontal Dirichlet values are prescribed at the corners; side Neumann stencils are imposed only at noncorner side nodes. The corner stencil defects above are reported, not imposed.",
    }


def free_term(point):
    px, py = point
    tolerance = 1e-12
    inside_closed = -tolerance <= px <= 2.0+tolerance and -tolerance <= py <= 1.0+tolerance
    if not inside_closed:
        return 0.0
    on_x = abs(px) < tolerance or abs(px-2.0) < tolerance
    on_y = abs(py) < tolerance or abs(py-1.0) < tolerance
    if on_x and on_y:
        return 0.25
    return 0.5 if on_x or on_y else 1.0


def _x_log_x(value):
    return 0.0 if value == 0.0 else value*np.log(value)


def _boundary_integral(sides, c, point, order):
    t, weight = _gauss(order)
    point = np.asarray(point)
    terms = {}
    for side in sides:
        tangent, normal, origin = side["tangent"], side["normal"], side["origin"]
        point_parameter = float((point-origin) @ tangent)
        signed_distance = float((point-origin) @ normal)
        left, right = float(side["nodes"][0]), float(side["nodes"][-1])
        on_side = abs(signed_distance) < 1e-12 and left-1e-12 <= point_parameter <= right+1e-12
        panels = list(map(float, side["nodes"]))
        if left < point_parameter < right and min(abs(node-point_parameter) for node in panels) > 1e-12:
            panels.append(point_parameter)
        panels = sorted(set(panels))
        result = 0.0
        if on_side:
            point_parameter = float(np.clip(point_parameter, left, right))
            singular_density = float(side["q"](point_parameter))
            a, b = point_parameter-left, right-point_parameter
            result -= singular_density*(_x_log_x(a)-a+_x_log_x(b)-b)/(2.0*np.pi)
        for a, b in zip(panels[:-1], panels[1:]):
            if b <= a:
                continue
            parameter = a+(b-a)*t
            xy = origin[None, :]+parameter[:, None]*tangent[None, :]
            difference = xy-point[None, :]
            radius = np.linalg.norm(difference, axis=1)
            if on_side:
                gn = np.zeros_like(radius)
            else:
                gn = normal_kernel(xy, point, normal, c)
            integrand = green_value(radius, c)*side["q"](parameter)-side["u"](parameter)*gn
            if on_side:
                integrand += np.log(radius)*singular_density/(2.0*np.pi)
            result += (b-a)*float(weight @ integrand)
        terms[side["name"]] = float(result)
    return float(sum(terms.values())), terms


def _volume_integral(c, point, order):
    t, weight = _gauss(order)
    point = np.asarray(point)
    if free_term(point) == 0.0:
        xx, yy = np.meshgrid(2.0*t, t)
        radius = np.hypot(xx-point[0], yy-point[1])
        return float(2.0*np.einsum("i,j,ij->", weight, weight, green_value(radius, c)*source_term(xx, yy, c)))
    vertices = np.array([[0., 0.], [2., 0.], [2., 1.], [0., 1.]])
    f_at_point = float(source_term(point[0], point[1], c))
    result = 0.0
    for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
        va, vb = a-point, b-point
        determinant = va[0]*vb[1]-va[1]*vb[0]
        if abs(determinant) < 1e-15:
            continue
        ray = (1.0-t[:, None])*va[None, :]+t[:, None]*vb[None, :]
        xy = point[None, None, :]+t[:, None, None]*ray[None, :, :]
        radius = np.linalg.norm(xy-point, axis=2)
        xx, yy = xy[:, :, 0], xy[:, :, 1]
        integrand = green_value(radius, c)*source_term(xx, yy, c)
        integrand += np.log(t)[:, None]*f_at_point/(2.0*np.pi)
        result += determinant*(float(np.einsum("i,j,ij->", weight*t, weight, integrand))+f_at_point/(8.0*np.pi))
    return float(result)


def evaluate_green_identity(x, y, u, c, point, order=64, exact_reference=False):
    x, y, u = _validate_grid(x, y, u)
    sides = build_boundary_traces(x, y, u, exact_reference=exact_reference)
    boundary, terms = _boundary_integral(sides, c, point, order)
    volume = _volume_integral(c, point, order)
    alpha = free_term(point)
    if exact_reference:
        point_u = float(exact_solution(*point))
    elif alpha == 0.0:
        point_u = None
    else:
        i = int(np.argmin(np.abs(x-point[0])))
        j = int(np.argmin(np.abs(y-point[1])))
        if abs(x[i]-point[0]) < 1e-12 and abs(y[j]-point[1]) < 1e-12:
            point_u = float(u[j, i])
        else:
            # Linear interpolation in x, then in y, also defines a bilinear
            # value for points which do not coincide with grid nodes.
            point_u = float(np.interp(point[1], y, [np.interp(point[0], x, row) for row in u]))
    lhs = 0.0 if alpha == 0.0 else alpha*point_u
    return {
        "c": int(c), "point": list(map(float, point)), "order": int(order),
        "trace_source": "exact reference only" if exact_reference else "finite differences with prescribed data exact",
        "alpha": float(alpha), "u_at_point": point_u,
        "lhs_alpha_u": float(lhs), "boundary_terms": terms,
        "boundary": boundary, "volume": volume,
        "rhs_boundary_minus_volume": float(boundary-volume),
        "residual_rhs_minus_lhs": float(boundary-volume-lhs),
    }


def audit_green_identities(x, y, u, c, orders=(16, 32, 64), include_exact_reference=True):
    if len(orders) < 2 or any(b <= a for a, b in zip(orders[:-1], orders[1:])):
        raise ValueError("At least two increasing quadrature orders are required")
    records = []
    for point in GREEN_POINTS:
        rows = [evaluate_green_identity(x, y, u, c, point, order) for order in orders]
        row = dict(rows[-1])
        row["quadrature_convergence"] = [
            {"order": value["order"], "boundary": value["boundary"],
             "volume": value["volume"], "rhs": value["rhs_boundary_minus_volume"],
             "residual": value["residual_rhs_minus_lhs"]}
            for value in rows
        ]
        row["quadrature_last_change"] = float(abs(rows[-1]["rhs_boundary_minus_volume"]-rows[-2]["rhs_boundary_minus_volume"]))
        if include_exact_reference:
            reference = [evaluate_green_identity(x, y, u, c, point, order, exact_reference=True) for order in orders]
            row["exact_trace_reference"] = reference[-1]
            row["exact_reference_last_change"] = float(abs(reference[-1]["rhs_boundary_minus_volume"]-reference[-2]["rhs_boundary_minus_volume"]))
        records.append(row)
    return {
        "c": int(c),
        "identity": "alpha*u = integral_boundary(G*q-u*G_n) - integral_domain(G*f); (Delta+c)G=-delta",
        "boundary_interpolation": "Piecewise linear interpolation of missing traces on each side. Prescribed Dirichlet/Neumann data are used analytically. Missing side u comes from U, missing top/bottom q from second-order one-sided y differences. The assigned nonexterior observation points coincide with grid nodes.",
        "volume_integration": "Analytic assigned f. Duffy triangles about interior/boundary observation point with constant-density log subtraction; ordinary tensor Gaussian quadrature for exterior point.",
        "boundary_integration": "Gaussian quadrature separately on every grid panel, split at the observation projection. The logarithmic singularity on the observation side is subtracted and integrated analytically; G_n=0 on the same flat side away from the observation point.",
        "quadrature_orders": list(map(int, orders)),
        "boundary_diagnostics": boundary_recovery_diagnostics(x, y, u),
        "checks": records,
        "max_quadrature_last_change": max(row["quadrature_last_change"] for row in records),
        "max_green_residual": max(abs(row["residual_rhs_minus_lhs"]) for row in records),
    }



def build_system(intervals, c):
    if intervals < 4 or c not in CASES:
        raise ValueError("Use at least four intervals and c in {-1,0,1}")
    x, y = np.linspace(0,2,intervals+1), np.linspace(0,1,intervals+1)
    hx, hy = x[1]-x[0], y[1]-y[0]
    ax, ay = 1/hx**2, 1/hy**2
    size = intervals-1
    rows, cols, values = [], [], []
    rhs = np.zeros(size*size)
    for j in range(1,intervals):
        for i in range(1,intervals):
            row = (j-1)*size+i-1
            a_minus = 0.0 if i == 1 else (2*ax/3 if i == intervals-1 else ax)
            a_plus = 0.0 if i == intervals-1 else (2*ax/3 if i == 1 else ax)
            diagonal = a_minus+a_plus+2*ay-c
            rhs[row] = -source_term(x[i],y[j],c)
            if a_minus:
                rows.append(row); cols.append(row-1); values.append(-a_minus)
            if a_plus:
                rows.append(row); cols.append(row+1); values.append(-a_plus)
            if i == intervals-1:
                rhs[row] += 8*y[j]/hx
            if j == 1:
                pass
            else:
                rows.append(row); cols.append(row-size); values.append(-ay)
            if j == intervals-1:
                rhs[row] += ay*(x[i]**3+1)
            else:
                rows.append(row); cols.append(row+size); values.append(-ay)
            rows.append(row); cols.append(row); values.append(diagonal)
    matrix = csr_matrix((values,(rows,cols)),shape=(size*size,size*size))
    return x,y,matrix,rhs


def solve_sor(intervals, c, omega=1.7, tolerance=1e-9, max_iterations=30000):
    x,y,matrix,rhs = build_system(intervals,c)
    diagonal = diags(matrix.diagonal(),format="csr")
    lower = tril(matrix,k=-1,format="csr")
    upper = triu(matrix,k=1,format="csr")
    left = (diagonal+omega*lower).tocsr()
    right = ((1-omega)*diagonal-omega*upper).tocsr()
    z = np.zeros_like(rhs)
    history = []
    for iteration in range(1,max_iterations+1):
        z = spsolve_triangular(left,omega*rhs+right@z,lower=True)
        residual = float(np.max(np.abs(matrix@z-rhs)))
        if iteration == 1 or iteration % 10 == 0 or residual <= tolerance:
            history.append((iteration,residual))
        if residual <= tolerance:
            break
    else:
        raise RuntimeError(f"SOR did not converge: N={intervals}, c={c}, residual={residual}")
    u = np.zeros((intervals+1,intervals+1))
    u[-1,:] = x**3+1
    u[1:-1,1:-1] = z.reshape((intervals-1,intervals-1))
    u[1:-1,0] = (4*u[1:-1,1]-u[1:-1,2])/3
    hx = x[1]-x[0]
    u[1:-1,-1] = (4*u[1:-1,-2]-u[1:-1,-3]+24*hx*y[1:-1])/3
    xx,yy = np.meshgrid(x,y)
    err = u-exact_solution(xx,yy)
    where = np.unravel_index(np.argmax(np.abs(err)),err.shape)
    laplacian = ((u[1:-1,2:]-2*u[1:-1,1:-1]+u[1:-1,:-2])/hx**2
                 +(u[2:,1:-1]-2*u[1:-1,1:-1]+u[:-2,1:-1])/(y[1]-y[0])**2)
    pde_residual = laplacian+c*u[1:-1,1:-1]-source_term(xx[1:-1,1:-1],yy[1:-1,1:-1],c)
    return x,y,u,{
        "c":int(c),"intervals_each_axis":int(intervals),"nodes_each_axis":int(intervals+1),
        "hx":float(hx),"hy":float(y[1]-y[0]),"interior_unknowns":int(z.size),
        "method":"SOR","omega":omega,"absolute_residual_tolerance":tolerance,
        "iterations":iteration,"algebraic_residual_inf":residual,
        "interior_PDE_residual_inf":float(np.max(np.abs(pde_residual))),
        "error_inf":float(np.max(np.abs(err))),"error_location":[float(x[where[1]]),float(y[where[0]])],
        "sor_history":history,"boundary":boundary_recovery_diagnostics(x,y,u),
    }


def save_figure(fig, directory, name):
    fig.savefig(directory/f"{name}.pdf")
    fig.savefig(directory/f"{name}.svg")
    fig.savefig(directory/f"{name}.png",dpi=300)
    plt.close(fig)


def plot_outputs(directory, solutions, runs, green):
    colors = {1:"#1c4f85",-1:"#ba4b20",0:"#333333"}
    styles = {1:"-",-1:"--",0:":"}
    x,y,u = solutions[1]
    fig,ax = plt.subplots(figsize=(6.934,2.25),layout="constrained")
    ax.plot([0,2,2,0,0],[0,0,1,1,0],color="#333333",lw=1.4)
    ax.text(1,-.17,r"$u = 0$",ha="center",fontsize=12)
    ax.text(1,1.1,r"$u = x^3+1$",ha="center",fontsize=12)
    ax.text(-.08,.48,r"$u_x = 0$",ha="right",fontsize=12)
    ax.text(2.08,.48,r"$u_x = 12y$",fontsize=12)
    ax.annotate("",xy=(2.34,0),xytext=(2,0),arrowprops={"arrowstyle":"->","lw":1.1,"color":"#333333"})
    ax.annotate("",xy=(0,1.23),xytext=(0,1),arrowprops={"arrowstyle":"->","lw":1.1,"color":"#333333"})
    ax.text(2.39,-.02,"x",fontsize=12,va="center")
    ax.text(-.03,1.26,"y",fontsize=12,ha="center")
    ax.text(-.04,-.04,"0",fontsize=10,ha="right",va="top")
    ax.text(2,-.04,"2",fontsize=10,ha="center",va="top")
    ax.text(-.04,1,"1",fontsize=10,ha="right",va="center")
    for px,py,label in ((1.1,.3,"M₁"),(1.1,-.3,"M₂"),(1.1,1,"M₃")):
        ax.plot(px,py,"o",color="#1c4f85",ms=4)
        offset = (6,-14) if py == 1 else (6,4)
        ax.annotate(label,(px,py),xytext=offset,textcoords="offset points",fontsize=12)
    ax.set(xlim=(-.50,2.62),ylim=(-.49,1.30),aspect="equal")
    ax.axis("off")
    save_figure(fig,directory,"task4_geometry")

    fig,axes = plt.subplots(2,1,figsize=(6.934,6.30),layout="constrained")
    for ys,color,ls in ((.25,"#1c4f85","-"),(.5,"#ba4b20","--"),(.75,"#333333",":")):
        j = int(np.argmin(abs(y-ys)))
        axes[0].plot(x,u[j],color=color,ls=ls,lw=1.7,label=f"МКР, y = {ys:g}".replace(".",","))
        axes[0].plot(x,exact_solution(x,ys),color=color,marker="o",ls="none",ms=3,markevery=2)
    axes[0].set(xlabel="x",ylabel="u",title="Решение при c = 1 по горизонтальным сечениям")
    axes[0].plot([],[],color="#333333",marker="o",ls="none",ms=3,label=r"Точное $u_*$")
    for c in CASES:
        xx,yy,uu = solutions[c]
        j = int(np.argmin(abs(yy-.5)))
        axes[1].plot(xx,uu[j]-exact_solution(xx,yy[j]),color=colors[c],ls=styles[c],lw=1.8,label=f"c = {c}")
    axes[1].set(xlabel="x",ylabel=r"$U-u_*$",title="Ошибка при y = 0,5")
    for ax in axes:
        ax.grid(alpha=.22); ax.legend(fontsize=10.5); ax.tick_params(labelsize=11)
        ax.xaxis.label.set_size(12); ax.yaxis.label.set_size(12); ax.title.set_fontsize(12.5)
    save_figure(fig,directory,"task4_solution_error")

    fig,axes = plt.subplots(2,1,figsize=(6.934,6.30),layout="constrained")
    for c in CASES:
        xx,yy,uu = solutions[c]
        sides = build_boundary_traces(xx,yy,uu)
        axes[0].plot(xx,sides[2]["values_q"]-(xx**3+3),color=colors[c],ls=styles[c],lw=1.8,label=f"c = {c}")
        axes[1].plot(yy,uu[:,-1]-exact_solution(2,yy),color=colors[c],ls=styles[c],lw=1.8,label=f"c = {c}")
    axes[0].set(xlabel="x",ylabel=r"$q_h-q_*$",title="Ошибка восстановленной производной на верхней стороне")
    axes[1].set(xlabel="y",ylabel=r"$U-u_*$",title="Ошибка решения на правой стороне")
    for ax in axes:
        ax.grid(alpha=.22); ax.legend(fontsize=10.5); ax.tick_params(labelsize=11)
        ax.xaxis.label.set_size(12); ax.yaxis.label.set_size(12); ax.title.set_fontsize(12.5)
    save_figure(fig,directory,"task4_boundary_errors")

    fig,axes = plt.subplots(2,1,figsize=(6.934,5.8),layout="constrained")
    for c in CASES:
        selected = [r for r in runs if r["c"]==c]
        hs = [r["hx"] for r in selected]
        axes[0].loglog(hs,[r["error_inf"] for r in selected],color=colors[c],ls=styles[c],marker="o",lw=1.6,label=f"c = {c}")
        axes[1].loglog(hs,[r["green_max_residual"] for r in selected],color=colors[c],ls=styles[c],marker="o",lw=1.6,label=f"c = {c}")
    axes[0].set(xlabel=r"$h_x$",ylabel=r"$E_\infty$",title="Сгущение сетки: ошибка решения")
    axes[1].set(xlabel=r"$h_x$",ylabel=r"$\max_m|R(M_m)|$",title="Сгущение сетки: невязка ГИУ")
    for ax in axes:
        ax.grid(alpha=.22,which="both"); ax.legend(fontsize=10.5); ax.tick_params(labelsize=11)
        ax.xaxis.label.set_size(12); ax.yaxis.label.set_size(12); ax.title.set_fontsize(12.5)
    save_figure(fig,directory,"task4_refinement")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=Path("task4_results"))
    args = parser.parse_args()
    directory = args.output_dir
    directory.mkdir(parents=True,exist_ok=True)
    canonical = canonical_reduction()
    runs, green_records, solutions = [], [], {}
    for n in (20,40,80):
        for c in CASES:
            x,y,u,run = solve_sor(n,c)
            green = audit_green_identities(x,y,u,c,include_exact_reference=(n==20))
            run["green_max_residual"] = green["max_green_residual"]
            runs.append(run)
            green["intervals_each_axis"] = n
            green_records.append(green)
            if n==20:
                solutions[c] = (x,y,u)
                xx,yy = np.meshgrid(x,y)
                np.savetxt(directory/f"task4_c{c:+d}.csv",np.column_stack((xx.ravel(),yy.ravel(),u.ravel(),exact_solution(xx,yy).ravel())),delimiter=",",header="x,y,U,exact",comments="")
            print(json.dumps({k:run[k] for k in ("c","intervals_each_axis","iterations","algebraic_residual_inf","error_inf","green_max_residual")}))
    result = {"canonical":canonical,"finite_differences":runs,"green":green_records}
    (directory/"task4_diagnostics.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf8")
    plot_outputs(directory,solutions,runs,green_records)
    print(f"Outputs written to {directory.resolve()}")


if __name__ == "__main__":
    main()
