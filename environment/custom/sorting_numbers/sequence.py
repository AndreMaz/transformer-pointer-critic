import numpy as np
import random

class Sequence():
    def __init__(self,
                min_value: int,
                max_value: int,
                seq_length: int,
                normalize: bool,
                add_special_chars: bool):

    # def generate(max_value, seq_length, normalize, add_special_chars):
        self.min_value = min_value
        self.max_value = max_value
        self.seq_length = seq_length
        self.normalize = normalize
        self.add_special_chars = add_special_chars

        ###### STATIC PLACEHOLDERS THAT SHOULD NOT BE MODIFIED ######
        self.MASK = 0
        self.__EOS_CODE = self.max_value + 1
        self.__SOS_CODE = self.max_value + 2
        self.vocab_size = self.max_value + 3 # MASK, SOS and EOS SYMBOLS
        
        # Generate a sorted sequence that will be modified later
        self.num_sequence = [num for num in range(self.min_value, self.max_value + 1)]

        if self.normalize:
            # This is a hack to leave the value "0" for self.MASK
            if (self.max_value % 2 == 1): raise ValueError('Max value must be even!')
            # Mean and Std are computed by including EOS_CODE and SOS_CODE
            self.num_mean, self.num_std = self.compute_stats()
            self.num_sequence = self.normalize_num_sequence()
            self.EOS = self.normalize_number(self.__EOS_CODE)
            self.SOS = self.normalize_number(self.__SOS_CODE)
        else: 
            self.EOS = self.__EOS_CODE
            self.SOS = self.__SOS_CODE

    def generate_sequence(self):
        # Shuffle the sequence
        random.shuffle(self.num_sequence)
        # Get a slice
        self.sequence = [self.EOS] + self.num_sequence[:self.seq_length]

        # Solution is a sorted sequence without EOS
        self.solution = list(self.sequence[1:])
        self.solution.sort()
        # Add SOS to the solution
        self.solution = [self.SOS] + self.solution
        
        # Build one-hot of actions
        self.correct_choices = np.zeros((self.seq_length + 1, self.seq_length + 1), dtype='int8')

        sorted_sequence = list(self.sequence[1:])
        sorted_sequence.sort()
        sorted_sequence = sorted_sequence + [self.EOS]

        for index, value in enumerate(sorted_sequence):
            # Get the index from "value" in unsorted input
            pos = self.sequence.index(value)
            # Set the one-hot
            self.correct_choices[index][pos] = 1
            
        if not self.add_special_chars:
            # Remove SOS, EOS chars
            self.sequence = self.sequence[1:]
            self.solution = self.solution[1:]

            # Remove last row that point for EOS
            self.correct_choices = np.delete(self.correct_choices, self.seq_length, 0)
            # Remove first colum reserved for EOS
            self.correct_choices = np.delete(self.correct_choices, 0, 1)

        # Return sequence, solution, correct choices
        return np.array(self.sequence.copy()), np.array(self.solution.copy()), np.copy(self.correct_choices)

    def compute_stats(self):
        copy_sequence = self.num_sequence.copy()
        # Add SOS and EOS. They also will be normalized
        copy_sequence.append(self.__EOS_CODE)
        copy_sequence.append(self.__SOS_CODE)

        copy_sequence = np.asarray(copy_sequence)
        # Compute mean and std
        num_mean = copy_sequence.mean()
        num_std = copy_sequence.std()
        return num_mean, num_std

    def normalize_num_sequence(self):
        normalized = [self.normalize_number(num) for num in self.num_sequence]
        return normalized

    def denormalize_num_sequence(self, sequence):
        denormalized = [int(self.denormalize_number(num)) for num in sequence]
        return denormalized

    def normalize_number(self, number):
        return (number - self.num_mean) / self.num_std
    
    def denormalize_number(self, number):
        return (number * self.num_std) + self.num_mean