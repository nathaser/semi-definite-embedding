import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import pickle
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D


def solve_SDP(D):
    n = len(D)
    col_ones = np.ones((n,1))

    X = cp.Variable((n,n), symmetric=True) 
    f = -cp.trace(X)
    obj = cp.Minimize(f)

    constraints = [
        col_ones.T @ X @ col_ones == 0,
        X >> 0
    ]
    for i in range(n):
        for j in range(i):
            constraints.append(X[i, i] + X[j, j] - 2 * X[i, j] == D[i, j])

    pb = cp.Problem(obj, constraints)
    pb.solve(verbose=True) # verbose=True to see the solver output (number of iterations, etc.)

    if pb.status != cp.OPTIMAL:
        print("Warning! Status:", pb.status)
    else:
        print("Problem solved. Optimal value f* = ", pb.value)

    return X.value

def post_and_plot(X):

    [v, V] = np.linalg.eig(X)
    print("The leading eigenvalues are: ", v[0:5])
    #print(V[0])
    x = np.sqrt(v[0]) * V[:,0]
    y = np.sqrt(v[1]) * V[:,1]
    z = np.sqrt(v[2]) * V[:,2]

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(x, y, z, 'C0')

    ax.axis('equal')
    plt.savefig('embedding3D.pdf', format='pdf', bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 16), v[:15], 'bo-', linewidth=2, markersize=8)
    plt.yscale('log') # log scale for better visibility of the spectrum decay
    plt.xlabel('eigenvalue index', fontsize=14)
    plt.ylabel('eigenvalue (Log scale)', fontsize=14)
    plt.title('Eigenvalue Spectrum of the Gram Matrix', fontsize=16)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig('eigenvalues_spectrum.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    
    return x,y,z

def solve_SDP_Bernoulli_sparsification(D, p):
    n = len(D)
    col_ones = np.ones((n,1))

    X = cp.Variable((n,n), symmetric=True) 
    f = -cp.trace(X)
    obj = cp.Minimize(f)

    constraints = [
        col_ones.T @ X @ col_ones == 0,
        X >> 0
    ]
    for i in range(n):
        for j in range(i):
            if np.random.binomial(n=1, p=p) == 1:
                constraints.append(X[i, i] + X[j, j] - 2 * X[i, j] == D[i, j])

    pb = cp.Problem(obj, constraints)
    pb.solve(verbose=True) # verbose=True to see the solver output (number of iterations, etc.)

    if pb.status != cp.OPTIMAL:
        print("Warning! Status:", pb.status)
    else:
        print("Problem solved. Optimal value f* = ", pb.value)

    return X.value

def prox_g(Z): # projection on the PSD cone
    eigvals, V = np.linalg.eigh(Z)
    eigvals = np.maximum(eigvals, 0)
    Z_proj = V @ np.diag(eigvals) @ V.T
    return (Z_proj + Z_proj.T) / 2.0 # to ensure symmetry

def prox_f(Z, D):
    n = Z.shape[0]    
    M = Z + np.eye(n)
    J = np.eye(n) - np.ones((n, n)) / n 
    M = J@M@J
    
    lam = np.zeros((n, n)) # initialize dual variables (Lagrange multipliers) to zero    
    alpha = 1.0 / (4.0*n)
    
    # dual ascent on the dual problem
    for _ in range(50):
        Astar_lam = np.diag(np.sum(lam, axis=1)) - lam 
        X = M - Astar_lam 
        
        A_X = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                A_X[i, j] = X[i, i] + X[j, j] - 2 * X[i, j]
        
        lam = lam + alpha*(A_X - D)
        
    # we go back to the primal solution at the end of the dual ascent loop
    Astar_lam = np.zeros((n, n))
    for i in range(n):
        somme_ligne = 0
        for k in range(n):
            somme_ligne += lam[i, k]
        for j in range(n):
            if i == j:
                Astar_lam[i, j] = somme_ligne - lam[i, i]
            else:
                Astar_lam[i, j] = -lam[i, j]
    
    X = M - Astar_lam
    
    return J@X@J # to ensure that the solution is centered, which is crucial for post_and_plot which uses np.sqrt() on the eigenvalues

def solve_SDP_douglas_rachford(D, eps=1e-4, maxiter=200):
    print("\nRunning Douglas-Rachford...")
    
    n = D.shape[0]
    Z = np.zeros((n, n))
    X_k = np.zeros((n, n))
    
    history_diff = []

    for k in range(maxiter):
        X_half = prox_f(Z, D)
        Y = prox_g(2 * X_half - Z)
        Z = Z + Y - X_half
        
        diff = np.linalg.norm(X_half - X_k, ord='fro')
        history_diff.append(diff)

        if diff < eps and k > 0:
            print(f"Convergence atteinte à l'itération {k} (diff = {diff:.6f})")
            X_k = X_half
            break
        
        if k % 10 == 0: # print diff every 10 iterations
            print(f"Iter {k}/{maxiter} - Différence: {diff:.6f}")
        
        X_k = X_half

    X_final = prox_g(X_k) # to ensure we end up in the PSD cone, which is crucial for post_and_plot which uses np.sqrt()
    
    # again to ensure non-negativity of eigenvalues, which is crucial for post_and_plot which uses np.sqrt()
    w, V = np.linalg.eigh(X_final) 
    w = np.maximum(w, 0.0)
    X_final = V @ np.diag(w) @ V.T 

    plt.figure(figsize=(8, 5))
    plt.plot(history_diff, 'b-', linewidth=2)
    plt.yscale('log')
    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel(r'$\|X_{k+1/2} - X_k\|_F$', fontsize=14)
    plt.title('Douglas-Rachford Convergence', fontsize=16)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig('dr_convergence.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    
    return (X_final + X_final.T) / 2.0 # to ensure symmetry

# fit a sphere to X,Y, and Z data points
# returns the radius and center points of the best fit sphere.

def sphereFit(spX,spY,spZ):
    # assemble the A matrix
    spX = np.array(spX)
    spY = np.array(spY)
    spZ = np.array(spZ)
    A = np.zeros((len(spX),4))
    A[:,0] = spX*2
    A[:,1] = spY*2
    A[:,2] = spZ*2
    A[:,3] = 1
    # assemble the f matrix
    f = np.zeros((len(spX),1))
    f[:,0] = (spX*spX) + (spY*spY) + (spZ*spZ)
    C, residules, rank, singval = np.linalg.lstsq(A,f)
    # solve for the radius
    t = (C[0]*C[0])+(C[1]*C[1])+(C[2]*C[2])+C[3]
    radius = np.sqrt(t)

    return radius, C[0], C[1], C[2]

with open('Distance_matrices.pkl', 'rb') as f:
    Dproto, D60, D30, D15, D = pickle.load(f)

# Xproto = solve_SDP(Dproto)
# post_and_plot(Xproto)

# X60 = solve_SDP(D60)
# post_and_plot(X60)

# X30 = solve_SDP(D30)
# post_and_plot(X30)

X = solve_SDP_douglas_rachford(D60, maxiter=200)
correctX, correctY, correctZ = post_and_plot(X)

rcParams['font.family'] = 'serif'

r, x0, y0, z0 = sphereFit(correctX,correctY,correctZ)

r = float(r)
print(f"\n=====================================")
print(f"Estimation du rayon terrestre : {r:.4f} * 1000 km")
print(f"Soit environ : {r * 1000:.0f} km")
print(f"=====================================\n")

u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*r
y = np.sin(u)*np.sin(v)*r
z = np.cos(v)*r
x = x + x0
y = y + y0
z = z + z0

# 3D plot of the Earth sphere
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(correctX, correctY, correctZ, zdir='z', s=20, c='b', rasterized=True, label='US Cities')

ax.plot_wireframe(x, y, z, color="r", alpha=0.3, label='Fitted Earth Sphere')

ax.set_aspect('equal')

limit = np.ceil(r) + 1
ax.set_xlim3d(-limit, limit)
ax.set_ylim3d(-limit, limit)
ax.set_zlim3d(-limit, limit)

ax.set_xlabel('$x$ (1000 km)', fontsize=14, labelpad=10)
ax.set_ylabel('$y$ (1000 km)', fontsize=14, labelpad=10)
zlabel = ax.set_zlabel('$z$ (1000 km)', fontsize=14, labelpad=10)
ax.set_title("3D Embedding and Earth Sphere Fitting", fontsize=16, pad=20)
ax.legend()

plt.savefig('earthFitted.pdf', format='pdf', dpi=300, bbox_extra_artists=[zlabel], bbox_inches='tight')
plt.show()