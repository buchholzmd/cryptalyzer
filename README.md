# Cryptalyzer
Genetic Algorithm to decrypt a simple substitution cipher. This code bank consists of four primary scripts and two auxiliary directories to store data and results, respectively. The following sections will explain the purpose and structure of the scripts in the codebase.

## Scripts

### corpus.py

This implements a Python class to represent a Corpus from a given text source. It can either be passed a list of strings containing the words in the corpus, or it can receive a list of urls to generate the corpus with. Stop words and non-alphabetical characters are removed from the text. Then ngram (n=1,2,3,4) statistics are computed. The ngrams with corresponding statistics (relative frequency, entropy and/or log probabilities) are stored in a pandas Dataframe. The dataframes are then saved to pickle files.

### cryptalyzer.py

This is simply a driver script to initialize the corpus and train the GA on four different ciphertext messages.

### genetic_algorithm.py

This script implements the Genetic Algorithm that searches key space for a cryptographic key that yields the most "English-like" decrypted plaintext (or whatever language the corpus is initialized with). The GA is trained to minimize the follwing objective function:

```
J(key) = -log(P{X=key})
```

The following explains the key components of the GA:

#### Initialization of the population

The population is initialized as a random sample (of user set self.size) from the search (key) space. The search (key) space is seeded by keys generated from the most frequent characters (monograms) in the ciphertext message. The key generation process is described in more detail in the section on the helper_functions.py script.

#### Selection for mating pool

The n keys with the smallest losses are used as the parent set for each epoch. n is a user set prefence (self.num_parents).

#### Crossover operator

Creates an offspring set (fixed size) from the parent set. Occurs with probability self.crossover_rate. The operator takes first half of key from parent 1 and second half from parent 2. Additionally another child will take first half of key from parent 2 and second half from parent 1. If an offspring has repeating characters, it is discarded and replaced with a random key from search (key) space. Finally, the random key is gauranteed to not be in the parent set, current offpsring set, or the previous population.

#### Mutation operator

Mutations occur with probability self.mutation_rate. The operator swaps two characters in a key, and gaurantees to not mutate the best key.

### helper_functions.py

This script contains various helper functions used in the Corpus object construction and in the GA for initialization, evaluation of the loss function, and the operators used during training. It contains functions to encrypt a message given a cryptographic key, decrypt a message, generate all cryptographic keys given an ngram distribution, calculate the entropy of ngrams, calculate the loss (negative log probabilities) of ngrams, create the distributions of ngrams (as pandas Dataframes), and to generate all permutations of a given string. The most important function is generate_keys(). It generates a cryptographic key for a substitution cipher by generating a key that would substitute for the most frequent ngrams (in the case monograms/characters), based upon the most frequent monograms in English ('etaoinsrhldcumfpgwybvkxjqz'). If k monograms are equiprobable, then multiple keys are generated, where the keys consist of all possible permutations of the equiprobable monograms.
