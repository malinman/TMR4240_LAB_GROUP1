
import numpy as np
from part_2.observer import NonlinearPassiveObserver

o = NonlinearPassiveObserver()

I = np.eye(3)
Z63 = np.zeros((6, 3))
Z33 = np.zeros((3, 3))

Ao = np.block([
    [o.Aw - o.K1 @ o.Cw, -o.K1, Z63],
    [-o.K2 @ o.Cw,        -o.K2, Z33],
    [-o.K3 @ o.Cw,        -o.K3, -o.Tb_inv]
])

Bo = np.vstack([
    np.zeros((6, 3)),
    I,
    Z33
])

Co = np.hstack([
    o.K4 @ o.Cw,
    o.K4,
    -I
])

eigenvalues = np.linalg.eigvals(Ao)

print("Maximum real eigenvalue:",
      np.max(eigenvalues.real))

print("Ao is Hurwitz:",
      np.all(eigenvalues.real < 0))

#Check necessary KYP condition for positive definite P
#Bo.T @ P = Co implies Co @ Bo = Bo.T @ P @ Bo

S = Co @ Bo

print("Co @ Bo =")
print(S)

print("Symmetric:", np.allclose(S, S.T))
print("Eigenvalues:", np.linalg.eigvalsh(0.5 * (S + S.T)))


#Frequency-domain check of H2
frequencies = np.logspace(-4, 2, 1000)
minimum_real_part = np.inf

for w in frequencies:
    H = Co @ np.linalg.solve(
        1j * w * np.eye(Ao.shape[0]) - Ao,
        Bo
    )

    hermitian_part = 0.5 * (H + H.conj().T)
    smallest_eigenvalue = np.linalg.eigvalsh(
        hermitian_part
    ).min()

    minimum_real_part = min(
        minimum_real_part,
        smallest_eigenvalue
    )

print("Minimum real part of H2:", minimum_real_part)
print("Positive on sampled frequencies:",
      minimum_real_part > 0)


import cvxpy as cp

n = Ao.shape[0]
P = cp.Variable((n, n), symmetric=True)
epsilon = cp.Variable()

Q = -(P @ Ao + Ao.T @ P)

problem = cp.Problem(
    cp.Maximize(epsilon),
    [
        P >> epsilon * np.eye(n),
        Q >> epsilon * np.eye(n),
        Bo.T @ P == Co,
        epsilon >= 0
    ]
)

problem.solve(solver=cp.CLARABEL)

print("KYP solver status:", problem.status)
print("KYP epsilon:", epsilon.value)

if P.value is not None:
    P_num = P.value
    Q_num = -(P_num @ Ao + Ao.T @ P_num)

    print("Minimum eigenvalue of P:",
          np.linalg.eigvalsh(P_num).min())
    print("Minimum eigenvalue of Q:",
          np.linalg.eigvalsh(Q_num).min())
    print("KYP equality residual:",
          np.linalg.norm(Bo.T @ P_num - Co))
