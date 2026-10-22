#!/usr/bin/env python3
"""Task 3: central Navier differences and direct traction equations.

Boundary derivatives use the lecture's two-point one-sided formula by default.
The optional --boundary-order 2 uses its three-point Taylor refinement.
Successive Gauss-Seidel updates are relaxed only in the interior. The stopping criterion
recomputes the actual PDE and traction equation residuals. Recovered reactions
and corner stresses are reported separately from this algebraic tolerance.
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
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import diags, eye, lil_matrix, tril, triu
from scipy.sparse.linalg import eigs, splu

plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix", "pdf.fonttype": 42})


def assemble(h=0.025, nu=0.3, boundary_order=1):
    if boundary_order not in (1,2):
        raise ValueError("boundary_order must be 1 or 2.")
    nx, ny = round(2/h), round(1/h)
    if nx < 2 or ny < 2 or abs(nx*h-2) > 1e-12 or abs(ny*h-1) > 1e-12:
        raise ValueError("h must divide both side lengths and give at least two intervals.")
    if not -1 < nu < 0.5:
        raise ValueError("Plane-strain Poisson ratio must lie in (-1,0.5).")
    x, y = np.linspace(0,2,nx+1), np.linspace(0,1,ny+1)
    A, C = (1-2*nu)/(2*(1-nu)), nu/(1-nu)
    B = A+C
    size = 2*(nx+1)*(ny+1)
    matrix, rhs = lil_matrix((size,size)), np.zeros(size)
    scale, interior = np.ones(size), np.zeros(size,dtype=bool)
    index = lambda i,j,c: 2*(j*(nx+1)+i)+c

    def add(row,i,j,c,value):
        matrix[row,index(i,j,c)] += value

    def first_difference(row,i,j,c,axis,factor):
        # Coefficients of h*D_x or h*D_y, including corner endpoints.
        position, end = (i,nx) if axis == 0 else (j,ny)
        if position == 0:
            terms = ((0,-1),(1,1)) if boundary_order == 1 else ((0,-1.5),(1,2),(2,-0.5))
        elif position == end:
            terms = ((end,1),(end-1,-1)) if boundary_order == 1 else ((end,1.5),(end-1,-2),(end-2,0.5))
        else:
            terms = ((position-1,-0.5),(position+1,0.5))
        for position,coefficient in terms:
            add(row,position if axis==0 else i,
                j if axis==0 else position,c,factor*coefficient)

    for j in range(ny+1):
        for i in range(nx+1):
            ru,rv = index(i,j,0),index(i,j,1)
            if i == 0:
                # Clamp takes priority at the two left corners.
                matrix[ru,ru] = matrix[rv,rv] = 1
            elif i == nx:
                # Right traction takes priority at the two right corners.
                first_difference(ru,i,j,0,0,1)
                first_difference(ru,i,j,1,1,C)
                first_difference(rv,i,j,0,1,1)
                first_difference(rv,i,j,1,0,1)
                rhs[rv] = h*y[j]*(1-y[j])/A
                scale[[ru,rv]] = 1/h
            elif j in (0,ny):
                first_difference(ru,i,j,0,1,1)
                first_difference(ru,i,j,1,0,1)
                first_difference(rv,i,j,1,1,1)
                first_difference(rv,i,j,0,0,C)
                scale[[ru,rv]] = 1/h
            else:
                interior[[ru,rv]] = True
                scale[[ru,rv]] = 1/h**2
                # Each row is minus h^2 times one Navier equation.
                for row,c,cx,cy in ((ru,0,1,A),(rv,1,A,1)):
                    add(row,i,j,c,2*(1+A))
                    add(row,i+1,j,c,-cx)
                    add(row,i-1,j,c,-cx)
                    add(row,i,j+1,c,-cy)
                    add(row,i,j-1,c,-cy)
                    for di,dj in ((1,1),(-1,-1),(1,-1),(-1,1)):
                        add(row,i+di,j+dj,1-c,-B*di*dj/4)
    matrix = matrix.tocsr()
    diagonal = matrix.diagonal()
    scale *= np.abs(diagonal)
    matrix = diags(1/diagonal) @ matrix
    rhs /= diagonal
    return x,y,matrix.tocsr(),rhs,scale,interior,{"A":A,"B":B,"C":C}


def gauss_seidel(matrix,rhs,scale,interior,epsilon=1e-5,omega=1.95,
                 initial=None,max_iterations=1500000):
    """Scalar sequential SOR, evaluated by lower triangular substitution.

Only the lower triangular sweep is factored and reused. The original system
is never factored or solved directly: this triangular solve is exactly the
ordinary sequence of scalar Gauss-Seidel updates.
    """
    if epsilon <= 0 or not 0 < omega < 2:
        raise ValueError("epsilon must be positive; omega must lie in (0,2).")
    relaxation = np.where(interior,omega,1.0)
    W = diags(relaxation)
    T = diags(np.ones(len(rhs)))+W@tril(matrix,k=-1,format="csr")
    R = diags(1-relaxation)-W@triu(matrix,k=1,format="csr")
    sweep = splu(T.tocsc(),permc_spec="NATURAL",diag_pivot_thresh=0)
    z = np.zeros_like(rhs) if initial is None else np.asarray(initial).copy()
    history = []
    for iteration in range(max_iterations+1):
        if iteration%100 == 0:
            residual = (matrix@z-rhs)*scale
            norm = float(np.max(np.abs(residual)))
            if iteration%1000 == 0 or norm <= epsilon:
                history.append([iteration,norm])
            if not np.isfinite(norm) or norm > 1e8:
                raise RuntimeError(f"Iteration diverged at sweep {iteration}.")
            if norm <= epsilon:
                return z,iteration,history
        if iteration != max_iterations:
            z = sweep.solve(relaxation*rhs+R@z)
    raise RuntimeError(f"True residual did not reach {epsilon:g} in {max_iterations} sweeps.")


def jacobi(matrix,rhs,scale,epsilon=1e-5,max_iterations=2500000):
    """Ordinary simultaneous updates, with the same physical residual as SOR."""
    z = np.zeros_like(rhs)
    history = []
    for iteration in range(max_iterations+1):
        if iteration%100 == 0 or iteration in (10,20,50):
            norm = float(np.max(np.abs((matrix@z-rhs)*scale)))
            if iteration%10000 == 0 or iteration in (10,20,50,100,200,300,500,1000,2000,5000) or norm <= epsilon:
                history.append([iteration,norm])
            if not np.isfinite(norm) or norm > 1e8:
                raise RuntimeError(f"Jacobi iteration diverged at sweep {iteration}.")
            if norm <= epsilon:
                return z,iteration,history
        if iteration != max_iterations:
            z += rhs-matrix@z  # assemble() normalizes the diagonal to one.
    raise RuntimeError(f"Jacobi did not reach {epsilon:g} in {max_iterations} sweeps.")


def compare_iterations(h,epsilon,omega,gauss_seidel_checks,boundary_order=1):
    """Optional numerical comparison; eigenvalues are estimates, not exact bounds."""
    x,y,matrix,rhs,scale,interior,op = assemble(h,boundary_order=boundary_order)
    J = (eye(len(rhs),format="csr")-matrix).tocsr()
    values,vectors = eigs(J,k=8,which="LM",ncv=80,tol=1e-10,
                         maxiter=12000,v0=np.linspace(.1,1,len(rhs)))
    rho = float(max(abs(values)))
    eigen_residual = max(float(np.linalg.norm(J@vectors[:,k]-values[k]*vectors[:,k],np.inf))
                         for k in range(len(values)))
    result = {"h":h,"boundary_order":boundary_order,
              "jacobi_spectral_radius_estimate":rho,
              "eigenpair_residual_inf":eigen_residual,
              "gauss_seidel_iterations":gauss_seidel_checks["iterations"],
              "gauss_seidel_residual_history":gauss_seidel_checks["residual_history"],
              "interior_relaxation":omega,"tolerance":epsilon}
    if rho < 1:
        z,iterations,history = jacobi(matrix,rhs,scale,epsilon)
        result.update(jacobi_iterations=iterations,jacobi_residual_history=history,
                      jacobi_residual_inf=float(np.max(np.abs((matrix@z-rhs)*scale))))
    else:
        result["jacobi_note"] = "Estimated spectral radius is not below one; no converged Jacobi result is claimed."
    return result


def recovered_stresses(U,V,h,A,C,boundary_order=1):
    uy,ux = np.gradient(U,h,h,edge_order=boundary_order)
    vy,vx = np.gradient(V,h,h,edge_order=boundary_order)
    return ux+C*vy,vy+C*ux,A*(uy+vx)


def reaction_estimates_from_sections(x,y,sigma_xx,sigma_xy,sections=(0.5,1.0,1.5)):
    """Estimate total support forces from computed interior section tractions.

    For the left subdomain 0 <= x <= a, the free horizontal edges carry no
    prescribed load. Its equilibrium gives Rx=-P(a), Ry=-Q(a), and
    MR=-M(a)-a*Q(a), where M(a)=-integral(y*sigma_xx(a,y),dy).
    The applied load is used only for the final error comparison. These
    estimates do not change or certify stresses recovered directly at x=0.
    """
    estimates = []
    for a in sections:
        i = int(np.argmin(abs(x-a)))
        if not np.isclose(x[i],a,rtol=0,atol=1e-12):
            raise ValueError("A reaction-check section must coincide with a grid column.")
        P = float(np.trapezoid(sigma_xx[:,i],y))
        Q = float(np.trapezoid(sigma_xy[:,i],y))
        M = -float(np.trapezoid(y*sigma_xx[:,i],y))
        Ry,MR = -Q,-M-a*Q
        estimates.append({
            "section_x":float(a),"normal_resultant":P,"shear_resultant":Q,
            "section_moment":M,"reaction_x":-P,"reaction_y":Ry,
            "reaction_moment_about_origin":MR,
            "force_balance_relative_error":abs(Ry+1/6)/(1/6),
            "moment_balance_relative_error":abs(MR+1/3)/(1/3),
        })
    return estimates


def transfer(previous,x,y):
    """Bilinear initial guess on a refined mesh; equations remain unchanged."""
    xx,yy,U,V = previous
    X,Y = np.meshgrid(x,y)
    points = np.column_stack((Y.ravel(),X.ravel()))
    z = np.empty(2*X.size)
    z[0::2] = RegularGridInterpolator((yy,xx),U)(points)
    z[1::2] = RegularGridInterpolator((yy,xx),V)(points)
    return z


def solve(h=0.025,nu=0.3,epsilon=1e-5,omega=1.95,previous=None,boundary_order=1):
    x,y,matrix,rhs,scale,interior,op = assemble(h,nu,boundary_order)
    z0 = None if previous is None else transfer(previous,x,y)
    z,iterations,history = gauss_seidel(matrix,rhs,scale,interior,epsilon,omega,z0)
    U = z[0::2].reshape(len(y),len(x))
    V = z[1::2].reshape(len(y),len(x))
    sx,sy,tau = recovered_stresses(U,V,h,op["A"],op["C"],boundary_order)
    resultants = [float(np.trapezoid(tau[:,i],y)) for i in range(len(x))]
    support_y = -resultants[0]
    support_x = -float(np.trapezoid(sx[:,0],y))
    support_moment = float(np.trapezoid(y*sx[:,0],y))
    residual = abs((matrix@z-rhs)*scale)
    mag = np.hypot(U,V)
    jmax,imax = np.unravel_index(np.argmax(mag),mag.shape)
    checks = {
        "mesh_spacing":h,"nodes":len(x)*len(y),"degrees_of_freedom":len(z),
        "method":f"central Navier differences; direct order-{boundary_order} traction equations",
        "boundary_derivative_order":boundary_order,
        "linear_solver":"scalar Gauss-Seidel with relaxed interior updates",
        "interior_relaxation":omega,"boundary_relaxation":1.0,
        "initial_guess":"zero" if previous is None else "bilinear transfer from preceding mesh",
        "iterations":iterations,"residual_history":history,
        "A":op["A"],"B":op["B"],"C":op["C"],"tolerance":epsilon,
        "true_residual_inf":float(residual.max()),
        "interior_navier_residual_inf":float(residual[interior].max()),
        "boundary_derivative_equation_residual_inf":float(residual[~interior].max()),
        "applied_force_y":1/6,"applied_moment_about_origin":1/3,
        "sampled_right_load_trapezoid":float(np.trapezoid(y*(1-y),y)),
        "recovered_clamp_reaction_x":support_x,
        "recovered_clamp_reaction_y":support_y,
        "recovered_clamp_moment_about_origin":support_moment,
        "recovered_clamp_reaction_relative_error":abs(support_y+1/6)/(1/6),
        "recovered_clamp_moment_relative_error":abs(support_moment+1/3)/(1/3),
        "recovered_shear_resultant_at_midlength":resultants[len(x)//2],
        "section_force_relative_error_at_midlength":abs(resultants[len(x)//2]-1/6)/(1/6),
        "section_moment_at_midlength":-float(np.trapezoid(y*sx[:,len(x)//2],y)),
        "reaction_estimates_from_sections":reaction_estimates_from_sections(x,y,sx,tau),
        "recovered_sigma_yy_at_right_corners":sy[[0,-1],-1].tolist(),
        "free_horizontal_sigma_yy_inf_excluding_corners":float(abs(sy[[0,-1],1:-1]).max()),
        "free_horizontal_sigma_xy_inf_excluding_corners":float(abs(tau[[0,-1],1:-1]).max()),
        "right_sigma_xx_inf_including_corners":float(abs(sx[:,-1]).max()),
        "right_sigma_xy_error_inf_including_corners":float(abs(tau[:,-1]-y*(1-y)).max()),
        "max_abs_u":float(abs(U).max()),"max_abs_v":float(abs(V).max()),
        "max_displacement":float(mag[jmax,imax]),
        "max_displacement_location":[float(x[imax]),float(y[jmax])],
        "symmetry_u_error":float(abs(U+U[::-1]).max()),
        "symmetry_v_error":float(abs(V-V[::-1]).max()),
        "corner_policy":"left corners: clamp; right corners: right-side traction equations",
        "accuracy_note":"Section-based total reactions are estimates from computed interior tractions and equilibrium of the left subdomain. They do not certify local clamp stresses. Direct clamp stress integrals and their larger errors remain separate.",
    }
    return x,y,U,V,sx,tau,resultants,checks


def make_figures(x, y, U, V, output_dir):
    # Conventional 2D line profiles; no filled contours, heatmaps or colorbars.
    fig, ax = plt.subplots(figsize=(6.934, 2.20), constrained_layout=True)
    ax.plot([0, 2, 2, 0, 0], [0, 0, 1, 1, 0], color='black', lw=1.5)
    for yy in np.linspace(0, 1, 11):
        ax.plot([-.07, 0], [yy-.045, yy], color='black', lw=.8)
    for yy in np.linspace(.10, .80, 6):
        ax.annotate('', xy=(2.04, yy + .35*yy*(1-yy)), xytext=(2.04, yy),
                    arrowprops={'arrowstyle': '->', 'color': '#1c4f85', 'lw': 1.3})
    ax.text(-.14, .50, r'$u = v = 0$', ha='right', va='center', fontsize=12)
    ax.text(1, 1.07, 'Свободный край', ha='center', fontsize=11)
    ax.text(1, -.20, 'Свободный край', ha='center', fontsize=11)
    ax.text(2.16, .52, r'$q(y) = y(1-y)$', fontsize=12, va='center')
    ax.annotate('', xy=(2.57, 0), xytext=(0, 0),
                arrowprops={'arrowstyle': '->', 'color': 'black', 'lw': 1.0})
    ax.annotate('', xy=(0, 1.20), xytext=(0, 0),
                arrowprops={'arrowstyle': '->', 'color': 'black', 'lw': 1.0})
    ax.text(2.64, -.02, 'x', fontsize=12, ha='center', va='top')
    ax.text(-.04, 1.22, 'y', fontsize=12, ha='right', va='bottom')
    ax.text(-.04, -.035, '0', fontsize=10, ha='right', va='top')
    ax.text(2.0, -.035, '2', fontsize=10, ha='center', va='top')
    ax.text(-.04, 1.0, '1', fontsize=10, ha='right', va='center')
    ax.set(xlim=(-.45, 2.70), ylim=(-.28, 1.24), aspect='equal')
    ax.axis('off')
    fig.savefig(output_dir / 'task3_geometry.pdf')
    fig.savefig(output_dir / 'task3_geometry.png', dpi=600)
    fig.savefig(output_dir / 'task3_geometry.svg')
    plt.close(fig)
    slices = (0.0, 0.25, 0.5)
    styles = (("#1c4f85", "-"), ("#ba4b20", "--"), ("#333333", ":"))
    for field, component in ((U, "u"), (V, "v")):
        fig, ax = plt.subplots(figsize=(6.934, 3.20), constrained_layout=True)
        for ys, (color, style) in zip(slices, styles):
            j = int(np.argmin(np.abs(y - ys)))
            ax.plot(x, field[j], color=color, ls=style, lw=1.9, label=f"y = {y[j]:g}".replace(".", ","))
        ax.set(xlim=(0, 2), xlabel="x", ylabel=component,
               title=f"{component}(x, y) на горизонтальных сечениях")
        ax.grid(True, alpha=0.22)
        ax.title.set_fontsize(12.5)
        ax.legend(fontsize=10.5)
        ax.tick_params(labelsize=11)
        ax.xaxis.label.set_size(12)
        ax.yaxis.label.set_size(12)
        fig.savefig(output_dir / f"task3_{component}_profiles.pdf")
        fig.savefig(output_dir / f"task3_{component}_profiles.png", dpi=600)
        fig.savefig(output_dir / f"task3_{component}_profiles.svg")
        plt.close(fig)
    sections = (0.5, 1.0, 2.0)
    for field, component in ((U, "u"), (V, "v")):
        fig, ax = plt.subplots(figsize=(6.934, 3.20), constrained_layout=True)
        for xs, (color, style) in zip(sections, styles):
            i = int(np.argmin(np.abs(x - xs)))
            ax.plot(y, field[:, i], color=color, ls=style, lw=1.9,
                    label=f"x = {x[i]:g}".replace(".", ","))
        ax.set(xlim=(0, 1), xlabel="y", ylabel=component,
               title=f"{component}(x, y) на вертикальных сечениях")
        ax.grid(True, alpha=0.22)
        ax.title.set_fontsize(12.5)
        ax.legend(fontsize=10.5)
        ax.tick_params(labelsize=11)
        ax.xaxis.label.set_size(12)
        ax.yaxis.label.set_size(12)
        fig.savefig(output_dir / f"task3_{component}_width_profiles.pdf")
        fig.savefig(output_dir / f"task3_{component}_width_profiles.png", dpi=600)
        fig.savefig(output_dir / f"task3_{component}_width_profiles.svg")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("task3_results"))
    parser.add_argument("--meshes", nargs="+", type=float, default=[0.025,0.0125,0.00625])
    parser.add_argument("--omega", type=float, default=1.95)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--boundary-order",type=int,choices=(1,2),default=1,
                        help="1: lecture boundary differences; 2: three-point Taylor refinement")
    parser.add_argument("--compare-iterations",action="store_true",
                        help="compare ordinary Jacobi with SOR on the first requested mesh")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    runs,previous = [],None
    for h in args.meshes:
        x,y,U,V,sx,tau,resultants,checks = solve(h,epsilon=args.epsilon,
            omega=args.omega,previous=previous,boundary_order=args.boundary_order)
        runs.append(checks)
        previous = (x,y,U,V)
        print(json.dumps(checks,ensure_ascii=False),flush=True)
        np.savez(args.output_dir/f"task3_grid_{h:g}.npz",x=x,y=y,U=U,V=V)
        if h == args.meshes[0]:
            with (args.output_dir/"task3_solution.csv").open("w",newline="",encoding="utf-8") as stream:
                writer=csv.writer(stream)
                writer.writerow(["x","y","U","V"])
                for j,yy in enumerate(y):
                    for i,xx in enumerate(x):
                        writer.writerow([xx,yy,U[j,i],V[j,i]])
            with (args.output_dir/"task3_clamp_reactions.csv").open("w",newline="",encoding="utf-8") as stream:
                writer=csv.writer(stream)
                writer.writerow(["y","recovered_traction_x","recovered_traction_y"])
                writer.writerows(zip(y,-sx[:,0],-tau[:,0]))
            with (args.output_dir/"task3_section_resultants.csv").open("w",newline="",encoding="utf-8") as stream:
                writer=csv.writer(stream)
                writer.writerow(["x","recovered_integral_sigma_xy_dy"])
                writer.writerows(zip(x,resultants))
            make_figures(x,y,U,V,args.output_dir)
            if args.compare_iterations:
                comparison = compare_iterations(h,args.epsilon,args.omega,checks,args.boundary_order)
                (args.output_dir/"task3_iteration_comparison.json").write_text(
                    json.dumps(comparison,ensure_ascii=False,indent=2),encoding="utf-8")
        summary={"model":"plane strain, scaled stress sigma/(lambda+2mu)",
                 "method":f"direct finite differences with order-{args.boundary_order} traction boundaries",
                 "load":"q(y)=y(1-y) on x=2","runs":runs}
        (args.output_dir/"task3_diagnostics.json").write_text(
            json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
