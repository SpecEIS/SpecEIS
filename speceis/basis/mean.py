import torch

class PolynomialMean:
    def __init__(self,degree=1):
        self.degree = degree

        if degree==1:
            self.n_dofs = 3
        elif degree==2:
            self.n_dofs = 6
        else:
            print('only 1 or 2 supported')

    def __call__(self,X,orthogonalize=False):
        if self.degree==1:
            h = torch.hstack((torch.ones(X.shape[0],1),X))
        else:
            h = torch.hstack((torch.ones(X.shape[0],1),X,X**2,(X[:,0]*X[:,1]).reshape(-1,1)))

        if orthogonalize:
            h,_ = torch.linalg.qr(h)

        return h
