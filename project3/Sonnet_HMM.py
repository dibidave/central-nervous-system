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
        Sonnet_Set.__init__(self, file_path, verbose=False)
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
            # print("Current sequence:", line_quantized)
            # print(N_syllables_left, "syllables left")
            if N_syllables_left>end_syllable_num_max:
                # print("We can't end the line now")
                # We can't end the line now
                state = random.choices(state_all, weights=self.A[states[-1]])[0]
                weights = [o if n<=N_syllables_left else 0 \
                           for o, n in zip(self.O[state], self._syllable_list_num_min)]
                obs = random.choices(obs_all, weights=weights)[0]
                n_syllables = random.choices(self._syllable_list_num_noend[obs])[0]
                if n_syllables>N_syllables_left:
                    continue
                states.append(state)
                line_quantized.append(obs)
                N_syllables.append(n_syllables)
                N_syllables_sum += n_syllables
            elif N_syllables_left>=end_syllable_num_min:
                # print("We can end the line now!")
                # We can end the line now!
                for state in state_all_shuffle:
                    # print("Trying state", state)
                    if self.A[states[-1]][state]==0:
                        # print("Impossible transition")
                        continue
                    weights = [self.O[state][k] if (flag and N_syllables_left in syl) \
                               else 0 for k, flag, syl in \
                               zip(obs_all, if_end_with, self._syllable_list_num)]
                    if sum(weights)==0:
                        # print("Impossible observation")
                        continue
                    obs = random.choices(obs_all, weights=weights)[0]
                    states.append(state)
                    line_quantized.append(obs)
                    N_syllables.append(N_syllables_left)
                    N_syllables_sum += N_syllables_left
                    break
            else:
                # print("No space for the last word, start over")
                # No space for the last word, start over
                states = states[0:1]
                line_quantized = line_quantized[0:1]
                N_syllables = N_syllables[0:1]
                N_syllables_sum = N_syllables[0]
        
        line = ' '.join([self._word_list[k] for k in line_quantized]).capitalize()
        
        if verbose:
            return line_quantized, line, states, N_syllables
        else:
            return line_quantized, line
        
    def generate_sonnet(self, N_lines_max=14, rhyme=True, verbose=False):
        states = []
        sonnet = []
        N_syllables = []
        
        if rhyme:
            N_lines_max=14
            for i in range(3):
                rhyme1_range = self._rhyme_dictionary.keys()
                line_quantized, line, states_k, N_syllables_k = \
                    self.generate_line_with_end(end_with=rhyme1_range, verbose=True)
                rhyme1 = line_quantized[-1]
                sonnet.append(line_quantized)
                states.append(states_k)
                N_syllables.append(N_syllables_k)
                
                rhyme2_range = self._rhyme_dictionary.keys()
                line_quantized, line, states_k, N_syllables_k = \
                    self.generate_line_with_end(end_with=rhyme2_range, verbose=True)
                rhyme2 = line_quantized[-1]
                sonnet.append(line_quantized)
                states.append(states_k)
                N_syllables.append(N_syllables_k)
                
                rhyme3_range = self._rhyme_pairs[self._rhyme_dictionary[rhyme1]].copy()
                rhyme3_range.remove(rhyme1)
                line_quantized, line, states_k, N_syllables_k = \
                    self.generate_line_with_end(end_with=rhyme3_range, verbose=True)
                sonnet.append(line_quantized)
                states.append(states_k)
                N_syllables.append(N_syllables_k)
        
                rhyme4_range = self._rhyme_pairs[self._rhyme_dictionary[rhyme2]].copy()
                rhyme4_range.remove(rhyme2)
                line_quantized, line, states_k, N_syllables_k = \
                    self.generate_line_with_end(end_with=rhyme4_range, verbose=True)
                sonnet.append(line_quantized)
                states.append(states_k)
                N_syllables.append(N_syllables_k)
            for i in range(1):
                rhyme1_range = self._rhyme_dictionary.keys()
                line_quantized, line, states_k, N_syllables_k = \
                    self.generate_line_with_end(end_with=rhyme1_range, verbose=True)
                rhyme1 = line_quantized[-1]
                sonnet.append(line_quantized)
                states.append(states_k)
                N_syllables.append(N_syllables_k)
                
                rhyme3_range = self._rhyme_pairs[self._rhyme_dictionary[rhyme1]].copy()
                rhyme3_range.remove(rhyme1)
                line_quantized, line, states_k, N_syllables_k = \
                    self.generate_line_with_end(end_with=rhyme3_range, verbose=True)
                sonnet.append(line_quantized)
                states.append(states_k)
                N_syllables.append(N_syllables_k)
        else:
            for k in range(N_lines_max):
                line_quantized, line, states_k, N_syllables_k = \
                    self.generate_line(verbose=True)
                sonnet.append(line_quantized)
                states.append(states_k)
                N_syllables.append(N_syllables_k)
        
        sonnet_string = self.print_sonnet(sonnet, sequence_type=Sequence_Type.LINE, 
                                          print_output=False)
        
        if verbose:
            return sonnet, sonnet_string, states, N_syllables
        else:
            return sonnet, sonnet_string

    def generate_haiku(self, N_syllables_line=(5, 7, 5), verbose=False):
        states = []
        haiku = []
        N_syllables = []
        haiku_string = ""
        
        for k in N_syllables_line:
            line_quantized, line, states_k, N_syllables_k = \
            self.generate_line(verbose=True, N_syllables_max=k)
            haiku.append(line_quantized)
            states.append(states_k)
            haiku_string += line+"\n"
            N_syllables.append(N_syllables_k)
        
        haiku_string = haiku_string[:-1]
        
        if verbose:
            return haiku, haiku_string, states, N_syllables
        else:
            return haiku, haiku_string
