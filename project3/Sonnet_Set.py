
from enum import Enum
from copy import deepcopy


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


class Sonnet_Set:

    # Skip any sonnets that are not this number of lines. Set to none if all
    # number of lines are allowed
    NUM_LINES_REQUIRED = 14

    NEW_LINE_CHARACTER = "\n"
    NEW_STANZA_CHARACTER = "_"

    RHYMING_PAIRS = [
        (0, 2),
        (1, 3),
        (4, 6),
        (5, 7),
        (8, 10),
        (9, 11),
        (12, 13)
    ]

    def __init__(self, file_path):

        current_sonnet = None
        sonnet_number = None

        sonnets = []
        self._sonnet_numbers = []

        with open(file_path, "r") as sonnets_file:

            for line in sonnets_file:
                line = line.strip()

                if current_sonnet is None:
                    if len(line) == 0:
                        continue
                    else:
                        try:
                            sonnet_number = int(line)
                            current_sonnet = []
                        except ValueError:
                            raise Exception("Failed to extract sonnet number, something is wrong")
                else:
                    if len(line) == 0:
                        if Sonnet_Set.NUM_LINES_REQUIRED is not None and \
                                len(current_sonnet) != Sonnet_Set.NUM_LINES_REQUIRED:
                            print("Sonnet %i is not %i lines, skipping" %
                                  (sonnet_number, Sonnet_Set.NUM_LINES_REQUIRED))
                        else:
                            sonnets.append(current_sonnet)
                            self._sonnet_numbers.append(sonnet_number)
                        current_sonnet = None
                        sonnet_number = None
                    else:
                        line_elements = line.split()
                        line_elements[-1] = line_elements[-1][0:-1]
                        print(line_elements)
                        current_sonnet.append(line_elements)

        self._sonnets_raw = sonnets

        self._word_dictionary = {}
        self._word_list = []

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
                sonnet_quantized.append(line_quantized)
            self._sonnets_quantized.append(sonnet_quantized)

        # We add a special word to indicate a new line
        self._word_dictionary[Sonnet_Set.NEW_LINE_CHARACTER] = len(self._word_list)
        self._word_list.append(Sonnet_Set.NEW_LINE_CHARACTER)

        # And another to indicate the end of a stanza
        self._word_dictionary[Sonnet_Set.NEW_STANZA_CHARACTER] = len(self._word_list)
        self._word_list.append(Sonnet_Set.NEW_STANZA_CHARACTER)

    def get_sequences(self, sequence_type=Sequence_Type.SONNET,
                      element_type=Element_Type.WORD):

        sequences = []

        if element_type == Element_Type.PHONEME:
            raise NotImplementedError()

        if sequence_type == Sequence_Type.SONNET:
            for sonnet in self._sonnets_quantized:
                sequence = []
                for line_index, line in enumerate(sonnet):
                    sequence.extend(line)
                    sequence.append(self._word_dictionary[Sonnet_Set.NEW_LINE_CHARACTER])
                    if (line_index + 1) % 4 == 0 or \
                            line_index == Sonnet_Set.NUM_LINES_REQUIRED - 1:
                        sequence.append(self._word_dictionary[Sonnet_Set.NEW_STANZA_CHARACTER])
                sequences.append(sequence)
        elif sequence_type == Sequence_Type.RHYMING_PAIR:
            for sonnet in self._sonnets_quantized:
                sequence = []

                for rhyming_pair in Sonnet_Set.RHYMING_PAIRS:

                    rhyming_pair_sequence = []

                    line_1 = sonnet[rhyming_pair[0]]
                    line_2 = sonnet[rhyming_pair[1]]

                    rhyming_pair_sequence.extend(line_1)
                    rhyming_pair_sequence.append(self._word_dictionary[Sonnet_Set.NEW_LINE_CHARACTER])
                    rhyming_pair_sequence.extend(line_2)

                    sequence.append(rhyming_pair_sequence)

                sequences.append(sequence)

        else:
            raise NotImplementedError()

        return sequences

    # Print a sonnet, given a certain sequence and element encoding scheme
    def print_sonnet(self, sequence, sequence_type=Sequence_Type.SONNET,
                     element_type=Element_Type.WORD):

        if element_type == Element_Type.PHONEME:
            raise NotImplementedError()

        sonnet_string = ""

        if sequence_type == Sequence_Type.SONNET:

            lines = []
            line = []

            for sequence_index, word_index in enumerate(sequence):

                word = self._word_list[word_index]

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

            sonnet_lines = [""] * len(Sonnet_Set.RHYMING_PAIRS) * 2

            for sequence_index, lines in enumerate(sequence):

                line_break_index = lines.index(
                    self._word_dictionary[Sonnet_Set.NEW_LINE_CHARACTER])

                line_1 = [self._word_list[x] for x in lines[0:line_break_index]]
                line_2 = [self._word_list[x] for x in lines[line_break_index + 1:]]

                sonnet_lines[Sonnet_Set.RHYMING_PAIRS[sequence_index][0]] = \
                    line_1

                sonnet_lines[Sonnet_Set.RHYMING_PAIRS[sequence_index][1]] = \
                    line_2

            sonnet_string = Sonnet_Set.convert_line_arrays_to_string(sonnet_lines)
        else:
            raise NotImplementedError()

        print(sonnet_string)

    @staticmethod
    def convert_line_arrays_to_string(lines):

        for line_index, line in enumerate(lines):
            lines[line_index] = " ".join(line)

        stanzas = []

        for line_index in range(0, 13, 4):
            stanzas.append(",\n".join(lines[line_index:line_index + 4]))

        sonnet_string = ":\n".join(stanzas) + "."

        return sonnet_string
