import random
import math

class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A=None, O=None, L=None, D=None, A_start=None, X=None, Y=None, train=True, N_iters=100, verbose=False):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0. 
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.
        Initialize in the following way:
        HiddenMarkovModel(A, O) returns an untrained HMM with defined matrices
        HiddenMarkovModel(L, D) returns an untrained HMM with defined parameters
        HiddenMarkovModel(X, Y) returns an supervised-trained HMM with trained matrices
        HiddenMarkovModel(X, L) returns an unsupervised-trained HMM with trained matrices

        Arguments:
            L:          Number of states.
            
            D:          Number of observations.
            
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.

            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.
            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to D - 1. In other words, a list of
                        lists.

            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to L - 1. In other words, a list of
                        lists.
                        
            train:      Whether if to train the model if the dataset is provided
        '''
        
        # Read the parameters from data, if provided
        self.X = X
        self.Y = Y
        if D is None and X is not None:
            observations = set()
            for x in X:
                observations |= set(x)
            D = len(observations)
        if L is None and Y is not None:
            states = set()
            for y in Y:
                states |= set(y)
            L = len(states)
        
        # Initialize either way
        if L is not None and D is not None:
            self.L = L
            self.D = D
            # Randomly initialize and normalize matrix A.
            A = [[random.random() for _ in range(L)] for _ in range(L)]
            A = [[j/sum(i) for j in i] for i in A]
            self.A = A
            # Randomly initialize and normalize matrix O.
            O = [[random.random() for _ in range(D)] for _ in range(L)]
            O = [[j/sum(i) for j in i] for i in O]
            self.O = O
        elif A is not None and O is not None:
            self.L = len(A)
            self.D = len(O[0])
            self.A = A
            self.O = O
        else:
            raise TypeError(''''Please initialize in the following way:
                HiddenMarkovModel(A, O) returns an untrained HMM with defined matrices
                HiddenMarkovModel(L, D) returns an untrained HMM with defined parameters
                HiddenMarkovModel(X, Y) returns an supervised-trained HMM with trained matrices
                HiddenMarkovModel(X, L) returns an unsupervised-trained HMM with trained matrices
                ''')
        
        # Initialize rest of the parameters
        if A_start is not None:
            self.A_start = A_start
        else:
            self.A_start = [1. / self.L for _ in range(self.L)]
        if self.A is None and self.O is None:
            self.A_log = [[math.log(x) if x>0 else -float('inf') for x in y] for y in self.A]
            self.O_log = [[math.log(x) if x>0 else -float('inf') for x in y] for y in self.O]
        self.A_start_log = [math.log(x) if x>0 else -float('inf') for x in self.A_start]
        
        # Train the model
        if train and X is not None:
            if Y is not None:
                self.supervised_learning(X, Y, verbose)
            else:
                self.unsupervised_learning(X, N_iters, verbose)
            
        

    def viterbi(self, x, verbose=False):
        '''
        Uses the Viterbi algorithm to find the max probability state 
        sequence corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            max_seq:    State sequence corresponding to x with the highest
                        probability.
        '''
        
        M = len(x)      # Length of sequence.

        # The (i, j)^th elements of probs and seqs are the max log probability
        # of the prefix of length i+1 ending in state j and the prefix
        # that gives this log probability, respectively.
        #
        # For instance, probs[0][0] is the log probability of the prefix of
        # length 1 ending in state 0.
        probs = [[-float('inf') for _ in range(self.L)] for _ in range(M)]
        seqs = [[[] for _ in range(self.L)] for _ in range(M)]

        for j in range(self.L):
            seqs[0][j] = [j]
            probs[0][j] = self.A_start_log[j]+self.O_log[j][x[0]]
            # print('Seq {} has log probability {:.3f}'.format(seqs[1][j], probs[1][j]))
        for i in range(1, M):
            for j in range(self.L):
                # To find the maximum probs[i][j], we have to compare across sequences resulting
                # from seqs[i-1][0...L](+)[j]
                # print('Comparing for sequence of length {} ending in {}'.format(i, j))
                best_idx = None
                for k in range(self.L):
                    prob = probs[i-1][k]+self.A_log[k][j]+self.O_log[j][x[i-1]]
                    # print('Seq {} has log probability {:.3f}'.format(seqs[i-1][k]+[j],
                    #                                              prob))
                    if prob>probs[i][j]:
                        # print('Better!')
                        probs[i][j] = prob
                        best_idx = k
                if best_idx is None:
                    seqs[i][j] = []
                else:
                    seqs[i][j] = seqs[i-1][best_idx]+[j]
                # print('Selected seq {} has log probability {:.3f}'.format(seqs[i][j],
                #                                              prob))
        
        final_prob = probs[M-1]
        max_prob = max(final_prob)
        max_idx = final_prob.index(max_prob)
        max_seq = seqs[M][max_idx]
        # max_seq_str = ''.join([str(xi) for xi in max_seq])
        if verbose:
            return max_seq, probs, seqs
        else:
            return max_seq


    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.

                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^0:i
                        and state y^i = j.

                        e.g. alphas[0][0] corresponds to the probability
                        of observing x^0:0, i.e. the first observation,
                        given that y^0 = 0, i.e. the first state is 0.
        '''

        M = len(x)      # Length of sequence.
        alphas = [[0. for _ in range(self.L)] for _ in range(M)]

        for j in range(self.L):
            alphas[0][j] = self.O[j][x[0]]*self.A_start[j]
        if normalize:
            sum_alphas = sum(alphas[0])
            alphas[0] = [x/sum_alphas for x in alphas[0]]
        for i in range(1, M): # Current position in the sequence
            for j in range(self.L): # Current state
                # To get the alphas[i][j], we have to sum across alphas resulting
                # from alphas[i-1][0...L]
                sum_ps = 0.
                for k in range(self.L): # Previous state
                    sum_ps += alphas[i-1][k]*self.A[k][j]
                alphas[i][j] = self.O[j][x[i-1]]*sum_ps
            if normalize:
                sum_alphas = sum(alphas[i])
                alphas[i] = [x/sum_alphas for x in alphas[i]]
        
        return alphas


    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.

                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing suffix x^i:M-1 and
                        state y^i = j.

                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M:M-1, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''

        M = len(x)      # Length of sequence.
        betas = [[0. for _ in range(self.L)] for _ in range(M)]

        for j in range(self.L):
            betas[M-1][j] = 1
        if normalize:
            sum_betas = sum(betas[M-1])
            betas[M-1] = [x/sum_betas for x in betas[M-1]]
        for i in range(M-2, -1, -1): # Current position in the sequence
            for j in range(self.L): # Current state
                # To get the betas[i][j], we have to sum across betas resulting
                # from betas[i+1][0...L]
                sum_ps = 0.
                for k in range(self.L): # Next state
                    sum_ps += betas[i+1][k]*self.A[j][k]*self.O[k][x[i]]
                betas[i][j] = sum_ps
            if normalize:
                sum_betas = sum(betas[i])
                betas[i] = [x/sum_betas for x in betas[i]]

        return betas
    
    
    def marginal(self, x):
        '''
        Uses the forward-backward algorithm to calculate the marginal
        probabilities of y corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            YO:         Marginal observation matrix with dimensions M x L.
                        The (i, j)^th element is P(y^i+1=j, x)

            YA:         Marginal transition matrix with dimensions M-1 x L x L.
                        The (i, j, k)^th element is P(y^i+1=j, y^i+2=k, x)

            YA_start:   Marginal starting transition matrix with dimensions L.
                        The (j)^th element is P(y^0=start, y^1=j, x)
        '''
        
        M = len(x)      # Length of sequence.
        alphas = self.forward(x, normalize=True)
        betas = self.backward(x, normalize=True)
        # alphas.pop(0) # Discard the first unused values, reduce the len to M
        # betas.pop(0) # Discard the first unused values, reduce the len to M
        YO = [[0. for _ in range(self.L)] for _ in range(M)]
        YA = [[[0. for _ in range(self.L)] for _ in range(self.L)] \
            for _ in range(M-1)]
        YA_start = [0. for _ in range(self.L)]
        
        # Calculate YA_start
        for i in range(1): # First position in the sequence
            sum_ya_start = 0.
            for j in range(self.L): # Current state
                YA_start[j] = betas[i][j]*self.A_start[j]*self.O[j][x[i]]
                sum_ya_start += YA_start[j]
            for j in range(self.L): # Current state
                YA_start[j] /= sum_ya_start
        
        # Calculate YA and YO
        for i in range(M-1): # Current position in the sequence
            sum_yo = 0.
            sum_ya = 0.
            for j in range(self.L): # Current state
                YO[i][j] = alphas[i][j]*betas[i][j]
                sum_yo += YO[i][j]
                for k in range(self.L): # Next state
                    YA[i][j][k] = alphas[i][j]*betas[i+1][k]*\
                        self.A[j][k]*self.O[k][x[i+1]]
                    sum_ya += YA[i][j][k]
            for j in range(self.L): # Current state
                YO[i][j] /= sum_yo
                for k in range(self.L): # Next state
                    YA[i][j][k] /= sum_ya
        for i in range(M-1, M): # Last position in the sequence
            sum_yo = 0.
            for j in range(self.L): # Current state
                YO[i][j] = alphas[i][j]*betas[i][j]
                sum_yo += YO[i][j]
            for j in range(self.L): # Current state
                YO[i][j] /= sum_yo
        
        return YO, YA, YA_start
        
    
    def supervised_learning(self, X, Y, verbose=False):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to D - 1. In other words, a list of
                        lists.

            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to L - 1. In other words, a list of
                        lists.

                        Note that the elements in X line up with those in Y.
        '''
        
        N = len(X)
        O_num = [[0. for _ in range(self.D)] for _ in range(self.L)]
        O_den = [0. for _ in range(self.L)]
        A_num = [[0. for _ in range(self.L)] for _ in range(self.L)]
        A_den = [0. for _ in range(self.L)]
        A_start_num = [0. for _ in range(self.L)]
        A_start_den = float(N)
        
        for i in range(N): # Input sequence
            M_i = len(X[i])
            # We observed start -> Y[i][0]
            A_start_num[Y[i][0]] += 1
            for j in range(M_i-1): # Token in the sequence
                # We observed Y[i][j] -> X[i][j] and Y[i][j] -> Y[i][j+1]
                O_den[Y[i][j]] += 1
                O_num[Y[i][j]][X[i][j]] += 1
                A_den[Y[i][j]] += 1
                A_num[Y[i][j]][Y[i][j+1]] += 1
            for j in range(M_i-1, M_i): # Last token in the sequence
                # We observed Y[i][j] -> X[i][j]
                O_den[Y[i][j]] += 1
                O_num[Y[i][j]][X[i][j]] += 1
        
        self.O = [[xi/y if y!=0 else float('nan') for xi in x] for x, y in zip(O_num, O_den)]
        self.A = [[xi/y if y!=0 else float('nan') for xi in x] for x, y in zip(A_num, A_den)]
        self.A_start = [xi/A_start_den for xi in A_start_num]
        
        if verbose:
            probs = 0.
            for i in range(len(X)): # Input sequence
                x = X[i]
                probs += math.log(self.probability_alphas(x))
            print('Training: current log probability = {:.4e}'.format(probs))
            return probs
        else:
            return None


    def unsupervised_learning(self, X, N_iters=100, verbose=False):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method updates the attributes of the HMM
        object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.

            N_iters:    The number of iterations to train on.
            
            verbose:    Flag whether to return the log likelihood of observing 
                        the data after each iteration
                        
        Returns:
            probs_iter  log likelihood of observing the data after each iteration
        '''
        
        if verbose:
            probs_iter = [0. for _ in range(N_iters)]
            probs_start = 0
            for i in range(len(X)): # Input sequence
                x = X[i]
                probs_start += math.log(self.probability_alphas(x))
            print('Training start: initial log probability = {:.4e}'.\
                  format(probs_start))
        
        for n_iter in range(N_iters):
            N = len(X)
            A_num = [[0. for _ in range(self.L)] for _ in range(self.L)]
            A_den = [0. for _ in range(self.L)]
            O_num = [[0. for _ in range(self.D)] for _ in range(self.L)]
            O_den = [0. for _ in range(self.L)]
            A_start_num = [0. for _ in range(self.L)]
            A_start_den = float(N)
            for n in range(N): # Input sequence
                x = X[n]
                M_n = len(x)
                YO, YA, YA_start = self.marginal(x)
                for j in range(self.L): # Possible state of the first token
                    # We observed start -> j with probability YA_start[j]
                    A_start_num[j] += YA_start[j]
                for i in range(M_n-1): # Token in the sequence
                    for j in range(self.L): # Possible state of the current token
                        # We observed j -> x[i] with probability YO[i][j]
                        O_den[j] += YO[i][j]
                        O_num[j][x[i]] += YO[i][j]
                        for k in range(self.L): # Possible state of the next token
                            # We observed j -> k with probability YA[i][j][k]
                            A_den[j] += YA[i][j][k]
                            A_num[j][k] += YA[i][j][k]
                for i in range(M_n-1, M_n): # Last token in the sequence
                    for j in range(self.L): # Possible state of the current token
                        # We observed j -> x[i] with probability YO[i][j]
                        O_den[j] += YO[i][j]
                        O_num[j][x[i]] += YO[i][j]
            self.O = [[xi/y if y!=0 else float('nan') for xi in x] \
                       for x, y in zip(O_num, O_den)]
            self.A = [[xi/y if y!=0 else float('nan') for xi in x] \
                       for x, y in zip(A_num, A_den)]
            self.A_start = [xi/A_start_den for xi in A_start_num]
            
            if verbose:
                for i in range(len(X)): # Input sequence
                    x = X[i]
                    probs_iter[n_iter] += math.log(self.probability_alphas(x))
                print('Training epoch {} of {}: current log probability = {:.4e}'.\
                      format(n_iter+1, N_iters, probs_iter[n_iter]), end='\r')
            else:
                print('Training epoch {} of {}.'.format(n_iter+1, N_iters), end='\r')
        
        print()
        if verbose:
            return probs_iter
        else:
            return None


    def generate_emission(self, M):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random. 

        Arguments:
            M:          Length of the emission to generate.

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''
        
        emission = [None for _ in range(M)]
        states = [None for _ in range(M)]
        obs_all = list(range(self.D))
        state_all = list(range(self.L))
        for i in range(0, 1):
            states[i] = random.choices(state_all, weights=self.A_start)[0]
            emission[i] = random.choices(obs_all, weights=self.O[states[i]])[0]
        for i in range(1, M):
            states[i] = random.choices(state_all, weights=self.A[states[i-1]])[0]
            emission[i] = random.choices(obs_all, weights=self.O[states[i]])[0]
        
        return emission, states


    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the state sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any state sequence, i.e.
        # the probability of x.
        prob = sum(alphas[-1])
        return prob


    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x)

        # beta_j(0) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = sum([betas[0][j] * self.A_start[j] * self.O[j][x[0]] \
                    for j in range(self.L)])

        return prob