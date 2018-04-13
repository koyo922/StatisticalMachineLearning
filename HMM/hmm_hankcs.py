# copyed from https://github.com/hankcs/hidden-markov-model/blob/master/hmm.py
# for reference & comparison

import numpy as np


class HMM:
    """
    Order 1 Hidden Markov Model
    Attributes
    ----------
    A : numpy.ndarray
        State transition probability matrix
    B: numpy.ndarray
        Output emission probability matrix with shape(N, number of output types)
    pi: numpy.ndarray
        Initial state probablity vector
    Common Variables
    ----------------
    obs_seq : list of int
        list of observations (represented as ints corresponding to output
        indexes in B) in order of appearance
    T : int
        number of observations in an observation sequence
    N : int
        number of states
    """

    def __init__(self, A, B, pi):
        self.A = A
        self.B = B
        self.pi = pi

    def _forward(self, obs_seq):
        N = self.A.shape[0]
        T = len(obs_seq)

        F = np.zeros((N, T))
        F[:, 0] = self.pi * self.B[:, obs_seq[0]]

        for t in range(1, T):
            for n in range(N):
                F[n, t] = np.dot(F[:, t - 1], (self.A[:, n])) * self.B[n, obs_seq[t]]

        return F

    def _backward(self, obs_seq):
        N = self.A.shape[0]
        T = len(obs_seq)

        X = np.zeros((N, T))
        X[:, -1:] = 1

        for t in reversed(range(T - 1)):
            for n in range(N):
                X[n, t] = np.sum(X[:, t + 1] * self.A[n, :] * self.B[:, obs_seq[t + 1]])

        return X

    def observation_prob(self, obs_seq):
        """ P( entire observation sequence | A, B, pi ) """
        return np.sum(self._forward(obs_seq)[:, -1])

    def state_path(self, obs_seq):
        """
        Returns
        -------
        V[last_state, -1] : float
            Probability of the optimal state path
        path : list(int)
            Optimal state path for the observation sequence
        """
        V, prev = self.viterbi(obs_seq)

        # Build state path with greatest probability
        last_state = np.argmax(V[:, -1])
        path = list(self.build_viterbi_path(prev, last_state))

        return V[last_state, -1], reversed(path)

    def viterbi(self, obs_seq):
        """
        Returns
        -------
        V : numpy.ndarray
            V [s][t] = Maximum probability of an observation sequence ending
                       at time 't' with final state 's'
        prev : numpy.ndarray
            Contains a pointer to the previous state at t-1 that maximizes
            V[state][t]
        """
        N = self.A.shape[0]
        T = len(obs_seq)
        prev = np.zeros((T - 1, N), dtype=int)

        # DP matrix containing max likelihood of state at a given time
        V = np.zeros((N, T))
        V[:, 0] = self.pi * self.B[:, obs_seq[0]]

        for t in range(1, T):
            for n in range(N):
                seq_probs = V[:, t - 1] * self.A[:, n] * self.B[n, obs_seq[t]]
                prev[t - 1, n] = np.argmax(seq_probs)
                V[n, t] = np.max(seq_probs)

        return V, prev

    def build_viterbi_path(self, prev, last_state):
        """Returns a state path ending in last_state in reverse order."""
        T = len(prev)
        yield (last_state)
        for i in range(T - 1, -1, -1):
            yield (prev[i, last_state])
            last_state = prev[i, last_state]

    def simulate(self, T, debug=False):
        if debug:
            O = [1, 0, 2, 2, 1, 2, 2, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 2, 2, 1, 0, 0, 2, 1, 2, 2, 0, 0, 0, 2, 2,
                 0, 0, 0, 1, 1, 2, 0, 2, 1, 0, 1, 1, 1, 2, 1, 0, 2, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 2, 2, 1, 2, 2, 1,
                 2, 2, 2, 0, 1, 0, 1, 1, 2, 2, 2, 1, 2, 0, 1, 2, 0, 2, 1, 1, 0, 1, 2, 0, 1, 1, 2, 0, 0, 0, 0, 2]
            S = [1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1,
                 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
            return np.array(O), np.array(S)

        def draw_from(probs):
            return np.where(np.random.multinomial(1, probs) == 1)[0][0]

        observations = np.zeros(T, dtype=int)
        states = np.zeros(T, dtype=int)
        states[0] = draw_from(self.pi)
        observations[0] = draw_from(self.B[states[0], :])
        for t in range(1, T):
            states[t] = draw_from(self.A[states[t - 1], :])
            observations[t] = draw_from(self.B[states[t], :])
        return observations, states

    def baum_welch_train(self, observations, criterion=0.05):
        n_states = self.A.shape[0]
        n_samples = len(observations)

        done = False
        while not done:
            # alpha_t(i) = P(O_1 O_2 ... O_t, q_t = S_i | hmm)
            # Initialize alpha
            alpha = self._forward(observations)

            # beta_t(i) = P(O_t+1 O_t+2 ... O_T | q_t = S_i , hmm)
            # Initialize beta
            beta = self._backward(observations)

            xi = np.zeros((n_states, n_states, n_samples - 1))
            for t in range(n_samples - 1):
                denom = np.dot(np.dot(alpha[:, t].T, self.A) * self.B[:, observations[t + 1]].T, beta[:, t + 1])
                for i in range(n_states):
                    numer = alpha[i, t] * self.A[i, :] * self.B[:, observations[t + 1]].T * beta[:, t + 1].T
                    xi[i, :, t] = numer / denom

            # gamma_t(i) = P(q_t = S_i | O, hmm)
            gamma = np.squeeze(np.sum(xi, axis=1))
            # Need final gamma element for new B
            prod = (alpha[:, n_samples - 1] * beta[:, n_samples - 1]).reshape((-1, 1))
            gamma = np.hstack((gamma, prod / np.sum(prod)))  # append one more to gamma!!!

            newpi = gamma[:, 0]
            newA = np.sum(xi, 2) / np.sum(gamma[:, :-1], axis=1).reshape((-1, 1))
            newB = np.copy(self.B)

            num_levels = self.B.shape[1]
            sumgamma = np.sum(gamma, axis=1)
            for lev in range(num_levels):
                mask = observations == lev
                newB[:, lev] = np.sum(gamma[:, mask], axis=1) / sumgamma

            if np.max(abs(self.pi - newpi)) < criterion and \
                    np.max(abs(self.A - newA)) < criterion and \
                    np.max(abs(self.B - newB)) < criterion:
                done = 1

            self.A[:], self.B[:], self.pi[:] = newA, newB, newpi


states = ('Healthy', 'Fever')

observations = ('normal', 'cold', 'dizzy')

start_probability = {'Healthy': 0.6, 'Fever': 0.4}

transition_probability = {
    'Healthy': {'Healthy': 0.7, 'Fever': 0.3},
    'Fever': {'Healthy': 0.4, 'Fever': 0.6},
}

emission_probability = {
    'Healthy': {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
    'Fever': {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6},
}


def generate_index_map(lables):
    index_label = {}
    label_index = {}
    i = 0
    for l in lables:
        index_label[i] = l
        label_index[l] = i
        i += 1
    return label_index, index_label


states_label_index, states_index_label = generate_index_map(states)
observations_label_index, observations_index_label = generate_index_map(observations)


def convert_observations_to_index(observations, label_index):
    list = []
    for o in observations:
        list.append(label_index[o])
    return list


def convert_map_to_vector(map, label_index):
    v = np.empty(len(map), dtype=float)
    for e in map:
        v[label_index[e]] = map[e]
    return v


def convert_map_to_matrix(map, label_index1, label_index2):
    m = np.empty((len(label_index1), len(label_index2)), dtype=float)
    for line in map:
        for col in map[line]:
            m[label_index1[line]][label_index2[col]] = map[line][col]
    return m


A = convert_map_to_matrix(transition_probability, states_label_index, states_label_index)
# print A
B = convert_map_to_matrix(emission_probability, states_label_index, observations_label_index)
# print B
observations_index = convert_observations_to_index(observations, observations_label_index)
pi = convert_map_to_vector(start_probability, states_label_index)
# print pi

h = HMM(A, B, pi)
V, p = h.viterbi(observations_index)
print(" " * 7, " ".join(("%10s" % observations_index_label[i]) for i in observations_index))
for s in range(0, 2):
    print("%7s: " % states_index_label[s] + " ".join("%10s" % ("%f" % v) for v in V[s]))
print('\nThe most possible states and probability are:')
p, ss = h.state_path(observations_index)
print(' '.join(states_index_label[s] for s in ss), end=' ')
print(p)

# run a baum_welch_train
observations_data, states_data = h.simulate(100, debug=True)
# print observations_data
# print states_data
guess = HMM(np.array([[0.5, 0.5],
                      [0.5, 0.5]]),
            np.array([[0.3, 0.3, 0.3],
                      [0.3, 0.3, 0.3]]),
            np.array([0.5, 0.5])
            )
guess.baum_welch_train(observations_data)
states_out = guess.state_path(observations_data)[1]
p = 0.0
for s in states_data:
    if next(states_out) == s: p += 1

print(p / len(states_data))
