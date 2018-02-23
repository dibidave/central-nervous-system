import numpy as np
np.seterr('raise')

def grad_U(Ui, Yij, Vj, reg, eta):
    """
    Takes as input Ui (the ith row of U), a training point Yij, the column
    vector Vj (jth column of V^T), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Ui multiplied by eta.
    """
    return eta*(reg*Ui-(Yij-np.dot(Ui, Vj))*Vj)   

def grad_V(Vj, Yij, Ui, reg, eta):
    """
    Takes as input the column vector Vj (jth column of V^T), a training point Yij,
    Ui (the ith row of U), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Vj multiplied by eta.
    """
    return eta*(reg*Vj-(Yij-np.dot(Ui, Vj))*Ui)   

def get_err(U, V, Y, reg=0.):
    """
    Takes as input a matrix Y of triples (i, j, Y_ij) where i is the index of a user,
    j is the index of a movie, and Y_ij is user i's rating of movie j and
    user/movie matrices U and V.

    Returns the mean regularized squared-error of predictions made by
    estimating Y_{ij} as the dot product of the ith row of U and the jth column of V^T.
    """
    err = 0.
    for i, j, Yij in Y.tolist():
        err+=(Yij-np.dot(U[:, i-1], V[:, j-1]))**2
    return reg/2*(np.linalg.norm(U)+np.linalg.norm(V))+1/2*err


def train_model(Y, M, N, K, eta=0.03, reg=0., eps=0.0001, max_epochs=300):
    """
    Given a training data matrix Y containing rows (i, j, Y_ij)
    where Y_ij is user i's rating on movie j, learns an
    K x M matrix U and K x N matrix V such that rating Y_ij is approximated
    by (U^TV)_ij.

    Uses a learning rate of <eta> and regularization of <reg>. Stops after
    <max_epochs> epochs, or once the magnitude of the decrease in regularized
    MSE between epochs is smaller than a fraction <eps> of the decrease in
    MSE after the first epoch.

    Returns a tuple (U, V, err) consisting of U, V, and the unregularized MSE
    of the model.
    """
    U = np.random.uniform(-0.5, 0.5, (K, M))
    V = np.random.uniform(-0.5, 0.5, (K, N))
    D = Y.shape[0] # Number of samples
    err0 = get_err(U, V, Y, reg)
    
    for epoch in range(max_epochs):
        print('Epoch', '{:>2d}'.format(epoch), end=': ')
        ids = np.random.permutation(D)
        for idx in ids:
            i = Y[idx, 0]
            j = Y[idx, 1]
            Yij = Y[idx, 2]
            Ui = U[:, i-1]
            Vj = V[:, j-1]
            dU = grad_U(Ui, Yij, Vj, reg, eta)
            dV = grad_V(Vj, Yij, Ui, reg, eta)
            U[:, i-1] = Ui-dU
            V[:, j-1] = Vj-dV
        if epoch==0:
            err_old = err0
            err = get_err(U, V, Y, reg)
            derr0 = err_old-err
        else:
            err_old = err
            err = get_err(U, V, Y, reg)
            derr = err_old-err
        print('current avarage training error', '{:.3f}'.format(err/D))
        if epoch>0 and derr<=derr0*eps:
            break
    return (U, V, get_err(U, V, Y, reg=0))



