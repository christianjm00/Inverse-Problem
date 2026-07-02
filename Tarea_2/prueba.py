def k(dim):
    K=np.zeros((dim,dim))
    n=np.arange(dim)
    K[n,n]=1/2
    K[n[1:],n[:-1]]=1/4
    K[n[:-1],n[1:]]=1/4
    return K