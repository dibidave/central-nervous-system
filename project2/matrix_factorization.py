import numpy as np
np.seterr('raise')

def grad_U(Ui, Vj, Yij, reg, eta, U_i_bias=0, V_j_bias=0, global_bias=0):
    """
    Takes as input Ui (the ith row of U), a training point Yij, the column
    vector Vj (jth column of V^T), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Ui multiplied by eta.
    """
    return eta*(reg*Ui-(Yij-global_bias-(np.dot(Ui, Vj)+U_i_bias+V_j_bias))*Vj)

def grad_V(Ui, Vj, Yij, reg, eta, U_i_bias=0, V_j_bias=0, global_bias=0):
    """
    Takes as input the column vector Vj (jth column of V^T), a training point Yij,
    Ui (the ith row of U), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Vj multiplied by eta.
    """
    return eta*(reg*Vj-(Yij-global_bias-(np.dot(Ui, Vj)+U_i_bias+V_j_bias))*Ui)

def grad_U_bias(Ui, Vj, Yij, reg, eta, U_i_bias=0, V_j_bias=0, global_bias=0):
    """
    Takes as input the column vector Vj (jth column of V^T), a training point Yij,
    Ui (the ith row of U), and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to the u_i, v_i bias terms
    """
    return eta*(reg*U_i_bias-(Yij-global_bias-(np.dot(Ui, Vj)+U_i_bias+V_j_bias)))

def grad_V_bias(Ui, Vj, Yij, reg, eta, U_i_bias=0, V_j_bias=0, global_bias=0):
    """
    Takes as input the column vector Vj (jth column of V^T), a training point Yij,
    Ui (the ith row of U), and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to the u_i, v_i bias terms
    """
    return eta*(reg*V_j_bias-(Yij-global_bias-(np.dot(Ui, Vj)+U_i_bias+V_j_bias)))

#def grad_global_bias(Ui, Vj, Yij, reg, eta, U_i_bias=0, V_j_bias=0, global_bias=0):
#    """
#    Takes as input the column vector Vj (jth column of V^T), a training point Yij,
#    Ui (the ith row of U), and eta (the learning rate).
#
#    Returns the gradient of the regularized loss function with
#        respect to the global bias terms
#    """
#    return eta*(-(Yij-global_bias-(np.dot(Ui, Vj)+U_i_bias+V_j_bias)))

def get_err(U, V, Y, reg=0., biases=None):
    """
    Takes as input a matrix Y of triples (i, j, Y_ij) where i is the index of a user,
    j is the index of a movie, and Y_ij is user i's rating of movie j and
    user/movie matrices U and V.

    Returns the mean regularized squared-error of predictions made by
    estimating Y_{ij} as the dot product of the ith row of U and the jth column of V^T.
    """
    err = 0.
    if biases is not None:
        for i, j, Yij in Y.tolist():
            if i>U.shape[1] or j>V.shape[1]:
                continue
            err += (Yij - (np.dot(U[:, i-1], V[:, j-1]) +
                           biases[0][i - 1] + biases[1][j - 1] + biases[2])) ** 2
        return reg/2*(np.linalg.norm(U)+np.linalg.norm(V)+
                      np.linalg.norm(biases[0])+np.linalg.norm(biases[1]))+1/2*err
    else:
        for i, j, Yij in Y.tolist():
            if i>U.shape[1] or j>V.shape[1]:
                continue
            err+=(Yij-np.dot(U[:, i-1], V[:, j-1]))**2
        return reg/2*(np.linalg.norm(U)+np.linalg.norm(V))+1/2*err


def train_model(Y, M, N, K, eta=0.03, reg=0., eps=0.0001, max_epochs=300,
                include_bias=False):
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

    if include_bias:
        biases = [
            np.random.uniform(-0.5, 0.5, (M, )),
            np.random.uniform(-0.5, 0.5, (N, )),
            np.nanmean(Y[:, 2])
        ]
    else:
        biases = None

    for epoch in range(max_epochs):
        print('Epoch', '{:>2d}'.format(epoch), end=': ')
        ids = np.random.permutation(D)
        for idx in ids:
            i = Y[idx, 0]
            j = Y[idx, 1]
            Yij = Y[idx, 2]
            Ui = U[:, i-1]
            Vj = V[:, j-1]
            if include_bias:
                dU = grad_U(Ui, Vj, Yij, reg, eta, biases[0][i - 1], biases[1][j - 1], biases[2])
                dV = grad_V(Ui, Vj, Yij, reg, eta, biases[0][i - 1], biases[1][j - 1], biases[2])
                delta_U_bias = grad_U_bias(Ui, Vj, Yij, reg, eta,
                                       biases[0][i - 1], biases[1][j - 1], biases[2])
                delta_V_bias = grad_V_bias(Ui, Vj, Yij, reg, eta,
                                       biases[0][i - 1], biases[1][j - 1], biases[2])
                biases[0][i - 1] -= delta_U_bias
                biases[1][j - 1] -= delta_V_bias
            else:
                dU = grad_U(Ui, Vj, Yij, reg, eta)
                dV = grad_V(Ui, Vj, Yij, reg, eta)
            U[:, i-1] = Ui-dU
            V[:, j-1] = Vj-dV
        if epoch==0:
            err_old = err0
            err = get_err(U, V, Y, reg, biases)
            derr0 = err_old-err
        else:
            err_old = err
            err = get_err(U, V, Y, reg, biases)
            derr = err_old-err
        
        print('current average training error', '{:.3f}'.format(err/D))
        if epoch>0 and derr<=derr0*eps:
            break

    if include_bias:
        return (U, V, biases, get_err(U, V, Y, biases=biases, reg=0))
    else:
        return (U, V, get_err(U, V, Y, biases=biases, reg=0))



