
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
                        current_sonnet.append(line.split())

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

            for sequence_index, word_index in enumerate(sequence):

                word = self._word_list[word_index]

                # Don't do anything for new stanza characters
                if word == Sonnet_Set.NEW_STANZA_CHARACTER:
                    continue

                sonnet_string += self._word_list[word_index]

                if sequence_index != len(sequence) - 1 \
                        and word != Sonnet_Set.NEW_LINE_CHARACTER:
                    sonnet_string += " "
        else:
            raise NotImplementedError()

        print(sonnet_string)
