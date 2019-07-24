import hdf5storage
import numpy as np
import scipy.sparse as sps
mat = hdf5storage.loadmat('prova.mat')

V = mat['defShape']
F = mat['faces']

def check_face_vertex(V, F):
	if V.shape[0] > V.shape[1]:
		V = V.T

	if F.shape[0] > F.shape[1]:
		F = F.T
	return V, F

V, F = check_face_vertex(V, F)
m, n = F.shape[1], V.shape[1]

i, j, s = np.hstack([F[0,:], F[1,:], F[2,:]]), np.hstack([F[1,:], F[2,:], F[0,:]]), np.hstack([range(m), range(m), range(m)])

kk = np.vstack([i,j]).transpose()
I, idx =np.unique(np.vstack([i,j]).transpose(), axis=0, return_index=True)
i, j, s = i[idx].astype(np.int), j[idx].astype(np.int), s[idx].astype(np.int)

S = sps.csr_matrix((s,(i,j)))
dnz = np.nonzero(S)
rnz = np.nonzero(S.transpose())
s1 = S[dnz[1],dnz[0]].A1
s2 = S[rnz[1],rnz[0]].A1

I = np.nonzero(np.logical_and(s1>0, s2>0))

E = np.stack([s1,s2]).transpose()
i = rnz[1][I]
j = rnz[0][I]

I = np.argwhere(i<j).squeeze()
E = E[I,:]

i = i[I]
j = j[I]

ne = i.shape[0]

e = V[:,j-1] - V[:,i-1]
d = np.sqrt(np.sum(e**2,0))
e = e /d
d = d / np.mean(d)
