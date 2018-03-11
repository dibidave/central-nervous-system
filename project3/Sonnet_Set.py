from enum import Enum
import re

# Different methods for representing sequences in sonnets.
class Sequence_Type(Enum):

    # A sequence is a complete sonnet. We add special characters for line
    # breaks and stanza breaks
    SONNET = 0

    # A sequence is a stanza. The stanza is separated by special characters for
    # line breaks
    STANZA = 1

    # A sequence is a rhyming pair - that is, each sonnet is broken up into
    # the pairs of lines that rhyme
    RHYMING_PAIR = 2

    # A sequence is a line, with no added special characters
    LINE = 3


# Different methods of encoding the data - words vs phonemes
class Element_Type(Enum):

    WORD = 1
    PHONEME = 2
    CHARACTER = 3


class Sonnet_Set:

    # Skip any sonnets that are not this number of lines. Set to none if all
    # number of lines are allowed
    NUM_LINES_REQUIRED = 14

    NEW_LINE_CHARACTER = "\n"
    NEW_STANZA_CHARACTER = "_"
    SPACE_CHARACTER = " "

    RHYMING_PAIRS = [
        (0, 2),
        (1, 3),
        (4, 6),
        (5, 7),
        (8, 10),
        (9, 11),
        (12, 13)
    ]

    def __init__(self, file_path="data/shakespeare.txt", verbose = False):

        current_sonnet = None
        sonnet_number = None

        sonnets = []
        self._sonnet_numbers = []

        with open(file_path, "r") as sonnets_file:

            for line in sonnets_file:
                line = line.strip("\r\n,.:;?!()'")

                if current_sonnet is None:
                    if len(line) == 0:
                        continue
                    else:
                        try:
                            sonnet_number = int(line)
                            current_sonnet = []
                        except ValueError:
                            raise Exception("Failed to extract sonnet \
                                number, something is wrong")
                else:
                    if len(line) == 0:
                        if Sonnet_Set.NUM_LINES_REQUIRED is not None and \
                                len(current_sonnet) != \
                                Sonnet_Set.NUM_LINES_REQUIRED:
                            print("Sonnet %i is not %i lines, skipping" %
                                  (sonnet_number,
                                   Sonnet_Set.NUM_LINES_REQUIRED))
                        else:
                            sonnets.append(current_sonnet)
                            self._sonnet_numbers.append(sonnet_number)
                        current_sonnet = None
                        sonnet_number = None
                    else:
                        # Convert to lowercase! Remove all commas, periods, question marks etc.!
                        line = line.lower()
                        line = re.sub("[,.:;?!()]", "", line)
                        # Remove ' at the begining and end of a word
                        line = line.replace(" '", " ")
                        line = line.replace("' ", " ")
                        # Restore some specific cases
                        line = re.sub(r"([\s^])t\b", r"\1t'", line)
                        line = re.sub(r"\bth\b", r"th'", line)
                        line = re.sub(r"\btis\b", r"'tis", line)
                        line = re.sub(r"\bgainst\b", r"'gainst", line)
                        line = re.sub(r"\btwixt\b", r"'twixt", line)
                        line = re.sub(r"\bscaped\b", r"'scaped", line)
                        line = re.sub(r"\bgreeing\b", r"'greeing", line)
                        
                        line_elements = line.split()
                        if "lovet'" in line_elements:
                            print(line)
                        if verbose: print(line_elements)
                        current_sonnet.append(line_elements)

        self._sonnets_raw = sonnets

        self._word_dictionary = {}
        self._word_list = []
        self._character_dictionary = {}
        self._character_list = []

        self._sonnets_quantized = []

        for sonnet in self._sonnets_raw:
            sonnet_quantized = []
            for line in sonnet:
                line_quantized = []
                for word in line:
                    if word not in self._word_dictionary:
                        self._word_dictionary[word] = len(self._word_list)
                        self._word_list.append(word)
                    word_index = self._word_dictionary[word]
                    line_quantized.append(word_index)
                    for character in word:
                        if character not in self._character_dictionary:
                            self._character_dictionary[character] = len(self._character_dictionary)
                            self._character_list.append(character)
                sonnet_quantized.append(line_quantized)
            self._sonnets_quantized.append(sonnet_quantized)

        # We add a special word to indicate a new line
        self._word_dictionary[Sonnet_Set.NEW_LINE_CHARACTER] = \
            len(self._word_list)
        self._word_list.append(Sonnet_Set.NEW_LINE_CHARACTER)
        
        self._character_dictionary[Sonnet_Set.NEW_LINE_CHARACTER] = \
            len(self._character_list)
        self._character_list.append(Sonnet_Set.NEW_LINE_CHARACTER)

        # And another to indicate the end of a stanza
        self._word_dictionary[Sonnet_Set.NEW_STANZA_CHARACTER] = \
            len(self._word_list)
        self._word_list.append(Sonnet_Set.NEW_STANZA_CHARACTER)
        
        self._character_dictionary[Sonnet_Set.NEW_STANZA_CHARACTER] = \
            len(self._character_list)
        self._character_list.append(Sonnet_Set.NEW_STANZA_CHARACTER)
        
        self._character_dictionary[Sonnet_Set.SPACE_CHARACTER] = \
            len(self._character_list)
        self._character_list.append(Sonnet_Set.SPACE_CHARACTER)
        
        # Add number of syllables
        if file_path=="data/shakespeare.txt":
            syllable_file_path = 'data/Syllable_dictionary.txt'
            self._syllable_dictionary = {}.fromkeys(self._word_list[:-2])
            with open(syllable_file_path, 'r') as syllable_file:
                for line in syllable_file:
                    line = line.strip().split(' ')
                    self._syllable_dictionary[line[0]] = line[1:]
            self._syllable_list = [self._syllable_dictionary[word] for word in self._word_list[:-2]]
            self._syllable_list_num =\
                [[int(n) if n.isdigit() else int(n[1:]) for n in l] for l in self._syllable_list]
            self._syllable_list_num_noend =\
                [[int(n) for n in l if n.isdigit()] for l in self._syllable_list]
            self._syllable_list_num_max = [max(l) for l in self._syllable_list_num]
            self._syllable_list_num_min = [min(l) for l in self._syllable_list_num]
        
        # Add rhyming words
        sonnet_sequences = self.get_sequences(Sequence_Type.RHYMING_PAIR, Element_Type.WORD)
        self._rhyme_dictionary = {}
        self._rhyme_pairs = []
        nl = self._word_dictionary[self.NEW_LINE_CHARACTER]
        for sonnet in sonnet_sequences:
            for rhyming_pair in sonnet:
                idx = rhyming_pair.index(nl)
                rhyme1 = rhyming_pair[idx-1]
                rhyme2 = rhyming_pair[-1]
                if rhyme1 not in self._rhyme_dictionary.keys()\
                    and rhyme2 not in self._rhyme_dictionary.keys():
                    self._rhyme_dictionary[rhyme1] = len(self._rhyme_pairs)
                    self._rhyme_dictionary[rhyme2] = len(self._rhyme_pairs)
                    self._rhyme_pairs.append([rhyme1, rhyme2])
                elif rhyme1 in self._rhyme_dictionary.keys()\
                    and rhyme2 in self._rhyme_dictionary.keys():
                    continue
                elif rhyme1 in self._rhyme_dictionary.keys():
                    idx = self._rhyme_dictionary[rhyme1]
                    self._rhyme_dictionary[rhyme2] = idx
                    self._rhyme_pairs[idx].append(rhyme2)
                else:
                    idx = self._rhyme_dictionary[rhyme2]
                    self._rhyme_dictionary[rhyme1] = idx
                    self._rhyme_pairs[idx].append(rhyme1)
        self._rhyme_pairs_string =\
            [[self._word_list[idx] for idx in pairs] for pairs in self._rhyme_pairs]
        self._if_word_rhymes = [1 if word in self._rhyme_dictionary.keys() else 0 \
                                for word in range(len(self._word_list)-2)]
    
    def get_sequences(self, sequence_type=Sequence_Type.SONNET,
                      element_type=Element_Type.WORD):

        sequences = []

        if element_type == Element_Type.PHONEME:
            raise NotImplementedError()
        elif element_type == Element_Type.CHARACTER:
            if sequence_type == Sequence_Type.SONNET:
                for sonnet in self._sonnets_quantized:
                    sequence = []
                    for line_index, line in enumerate(sonnet):
                        for word_index, word_number in enumerate(line):
                            word = self._word_list[word_number]
                            for character in word:
                                sequence.append(self._character_dictionary[character])
                            if word_index != len(line) - 1:
                                sequence.append(self._character_dictionary[Sonnet_Set.SPACE_CHARACTER])
                        sequence.append(self._character_dictionary[Sonnet_Set.NEW_LINE_CHARACTER])
                    sequences.append(sequence)
        elif sequence_type == Sequence_Type.SONNET:
            for sonnet in self._sonnets_quantized:
                sequence = []
                for line_index, line in enumerate(sonnet):
                    sequence.extend(line)
                    sequence.append(
                        self._word_dictionary[Sonnet_Set.NEW_LINE_CHARACTER])
                    if (line_index + 1) % 4 == 0 or \
                            line_index == Sonnet_Set.NUM_LINES_REQUIRED - 1:
                        sequence.append(
                            self._word_dictionary[
                                Sonnet_Set.NEW_STANZA_CHARACTER])
                sequences.append(sequence)
        elif sequence_type == Sequence_Type.RHYMING_PAIR:
            for sonnet in self._sonnets_quantized:
                sequence = []

                for rhyming_pair in Sonnet_Set.RHYMING_PAIRS:

                    rhyming_pair_sequence = []

                    line_1 = sonnet[rhyming_pair[0]]
                    line_2 = sonnet[rhyming_pair[1]]

                    rhyming_pair_sequence.extend(line_1)
                    rhyming_pair_sequence.append(
                        self._word_dictionary[Sonnet_Set.NEW_LINE_CHARACTER])
                    rhyming_pair_sequence.extend(line_2)

                    sequence.append(rhyming_pair_sequence)

                sequences.append(sequence)
        elif sequence_type == Sequence_Type.LINE:

            sequences = []

            for sonnet in self._sonnets_quantized:
                lines = []

                for line in sonnet:
                    lines.append(line[:])
                sequences.append(lines)
        else:
            raise NotImplementedError()

        return sequences

    # Print a sonnet, given a certain sequence and element encoding scheme
    def print_sonnet(self, sequence, sequence_type=Sequence_Type.SONNET,
                     element_type=Element_Type.WORD, print_output=True):
        
        sonnet_string = ""

        if element_type == Element_Type.PHONEME:
            raise NotImplementedError()
        elif element_type == Element_Type.CHARACTER:
            if sequence_type == Sequence_Type.SONNET:
                is_new_line = True
                line_index = 0
                for character_index in sequence:
                    character = self._character_list[character_index]
                    if is_new_line:
                        character = character.upper()
                        if line_index in [12, 13]:
                            sonnet_string += "  "
                        is_new_line = False
                    if character == Sonnet_Set.NEW_LINE_CHARACTER:
                        is_new_line = True
                        if line_index in [3, 7, 11]:
                            sonnet_string += ":"
                        elif line_index == 13:
                            sonnet_string += "."
                        else:
                            sonnet_string += ","
                        line_index += 1
                    sonnet_string += character
            else:
                raise NotImplementedError()
        elif sequence_type == Sequence_Type.SONNET:

            lines = []
            line = []

            for sequence_index, word_index in enumerate(sequence):

                word = self._word_list[word_index]
                # Capitalize if it's the first word of a line
                if not line:
                    word = word.capitalize()
                
                # Don't do anything for new stanza characters
                if word == Sonnet_Set.NEW_STANZA_CHARACTER:
                    continue
                elif word == Sonnet_Set.NEW_LINE_CHARACTER:
                    lines.append(line)
                    line = []
                else:
                    line.append(self._word_list[word_index])

            sonnet_string = Sonnet_Set.convert_line_arrays_to_string(lines)

        elif sequence_type == Sequence_Type.RHYMING_PAIR:

            print(sequence)
            print("Converts to...")

            sonnet_lines = [""] * len(Sonnet_Set.RHYMING_PAIRS) * 2

            for sequence_index, lines in enumerate(sequence):

                try:
                    line_break_index = lines.index(
                        self._word_dictionary[Sonnet_Set.NEW_LINE_CHARACTER])

                    line_1 = [self._word_list[x]
                              for x in lines[0:line_break_index]]
                    line_2 = [self._word_list[x]
                              for x in lines[line_break_index + 1:]]
                except ValueError:
                    midpoint = int(len(lines)/2)
                    line_1 = [self._word_list[x] for x in lines[0:midpoint]]
                    line_2 = [self._word_list[x] for x in lines[midpoint:]]

                sonnet_lines[Sonnet_Set.RHYMING_PAIRS[sequence_index][0]] = \
                    line_1

                sonnet_lines[Sonnet_Set.RHYMING_PAIRS[sequence_index][1]] = \
                    line_2

            sonnet_string = Sonnet_Set.convert_line_arrays_to_string(
                sonnet_lines)
        elif sequence_type == Sequence_Type.LINE:

            lines = []

            for line in sequence:
                lines.append([self._word_list[x] for x in line])
            sonnet_string = Sonnet_Set.convert_line_arrays_to_string(lines)
        else:
            raise NotImplementedError()
        
        if print_output:
            print(sonnet_string)
        return sonnet_string

    @staticmethod
    def convert_line_arrays_to_string(lines):

        for line_index, line in enumerate(lines):
            line_string = " ".join(line)
            line_string = line_string[0].upper()+line_string[1:] # Capitalize!
            lines[line_index] = line_string
            if line_index in [12, 13]:
                lines[line_index] = "  "+lines[line_index]

        stanzas = []

        for line_index in range(0, 13, 4):
            stanzas.append(",\n".join(lines[line_index:line_index + 4]))

        sonnet_string = ":\n".join(stanzas) + "."

        return sonnet_string
