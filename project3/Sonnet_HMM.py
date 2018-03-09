from Sonnet_Set import Sonnet_Set
from Sonnet_Set import Sequence_Type
from Sonnet_Set import Element_Type
from HMM_JL import HiddenMarkovModel as HMM
import itertools
import random

class Sonnet_HMM(HMM, Sonnet_Set):
    '''
    HMM for Shakespearean sonnets
    '''
    
    def __init__(self, file_path="data/shakespeare.txt", N_state=8, N_iters=100, 
                 train=True, verbose=False):
        self.N_state = N_state
        self.N_iters = N_iters
        Sonnet_Set.__init__(self, file_path, verbose)
        self.sonnet_sequences = self.get_sequences(Sequence_Type.LINE, Element_Type.WORD)
        self.sonnet_lines = list(itertools.chain(*self.sonnet_sequences))
        self.N_observation = len(self._word_list)-2
        HMM.__init__(self, L=self.N_state, D=self.N_observation, X=self.sonnet_lines, 
                     train=train, N_iters=N_iters, verbose=verbose)
    
    def unsupervised_learning(self, X=None, N_iters=100, verbose=False):
        if X is None:
            return HMM.unsupervised_learning(self, X=self.X, N_iters=N_iters, verbose=verbose)
        else:
            return HMM.unsupervised_learning(self, X=X, N_iters=N_iters, verbose=verbose)
        
    def generate_line(self, N_syllables_max=10, verbose=False):
        states = []
        line_quantized = []
        N_syllables = []
        obs_all = list(range(self.D))
        state_all = list(range(self.L))
        
        state = random.choices(state_all, weights=self.A_start)[0]
        obs = random.choices(obs_all, weights=self.O[state])[0]
        n_syllables = self._syllable_list_num_noend[obs][0]
        states.append(state)
        line_quantized.append(obs)
        N_syllables.append(n_syllables)
        N_syllables_sum = n_syllables
        while N_syllables_sum<N_syllables_max:
            N_syllables_left = N_syllables_max-N_syllables_sum
            state = random.choices(state_all, weights=self.A[states[-1]])[0]
            weights = [o if n<=N_syllables_left else 0 \
                       for o, n in zip(self.O[state], self._syllable_list_num_min)]
            obs = random.choices(obs_all, weights=weights)[0]
            if N_syllables_left in self._syllable_list_num[obs]:
                # We can end the line now!
                states.append(state)
                line_quantized.append(obs)
                N_syllables.append(N_syllables_left)
                break
            else:
                # Current word is too short to end the line
                n_syllables = random.choices(self._syllable_list_num_noend[obs])[0]
                states.append(state)
                line_quantized.append(obs)
                N_syllables.append(n_syllables)
                N_syllables_sum += n_syllables
        
        line = ' '.join([self._word_list[k] for k in line_quantized]).capitalize()
        
        if verbose:
            return line_quantized, line, states, N_syllables
        else:
            return line_quantized, line
        
    def generate_line_with_end(self, N_syllables_max=10, verbose=False, end_with=None):
        states = []
        line_quantized = []
        N_syllables = []
        obs_all = list(range(self.D))
        state_all = list(range(self.L))
        state_all_shuffle = state_all.copy()
        random.shuffle(state_all_shuffle)
        if end_with is None:
            end_with = obs_all
        if_end_with = [1 if obs in end_with else 0 for obs in obs_all]
        
        end_syllable_list_num_min = [self._syllable_list_num_min[k] for k in end_with]
        end_syllable_list_num_max = [self._syllable_list_num_max[k] for k in end_with]
        end_syllable_num_min = min(end_syllable_list_num_min)
        end_syllable_num_max = max(end_syllable_list_num_max)
        
        state = random.choices(state_all, weights=self.A_start)[0]
        obs = random.choices(obs_all, weights=self.O[state])[0]
        n_syllables = self._syllable_list_num_noend[obs][0]
        states.append(state)
        line_quantized.append(obs)
        N_syllables.append(n_syllables)
        N_syllables_sum = n_syllables
        while N_syllables_sum<N_syllables_max:
            N_syllables_left = N_syllables_max-N_syllables_sum
            if N_syllables_left in list(range(end_syllable_num_min, end_syllable_num_max+1)):
                # We can end the line now!
                for state in state_all_shuffle:
                    if self.A[states[-1]][state]==0:
                        continue
                    weights = [self.O[state][k] if (flag and N_syllables_left in syl) else 0 \
                               for k, flag, syl in zip(obs_all, if_end_with, self._syllable_list_num)]
                    if sum(weights)==0:
                        continue
                    obs = random.choices(obs_all, weights=weights)[0]
                    states.append(state)
                    line_quantized.append(obs)
                    N_syllables.append(N_syllables_left)
            else:
                # Current word is too short to end the line
                state = random.choices(state_all, weights=self.A[states[-1]])[0]
                weights = [o if n<=N_syllables_left else 0 \
                           for o, n in zip(self.O[state], self._syllable_list_num_min)]
                obs = random.choices(obs_all, weights=weights)[0]
                n_syllables = random.choices(self._syllable_list_num_noend[obs])[0]
                states.append(state)
                line_quantized.append(obs)
                N_syllables.append(n_syllables)
                N_syllables_sum += n_syllables
        
        line = ' '.join([self._word_list[k] for k in line_quantized]).capitalize()
        
        if verbose:
            return line_quantized, line, states, N_syllables
        else:
            return line_quantized, line
        
    def generate_sonnet(self, N_lines_max=14, verbose=False):
        states = []
        sonnet = []
        N_syllables = []
        
        for k in range(N_lines_max):
            line_quantized, line, states_k, N_syllables_k = self.generate_line(verbose=True)
            sonnet.append(line_quantized)
            states.append(states_k)
            N_syllables.append(N_syllables_k)
        
        sonnet_string = self.print_sonnet(sonnet, sequence_type=Sequence_Type.LINE, 
                                          print_output=False)
        
        if verbose:
            return sonnet, sonnet_string, states, N_syllables
        else:
            return sonnet, sonnet_string
