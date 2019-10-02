import os
import time
import heapq
import random

from helper_functions import *
##############################
##
# Author: Matthew Buchholz
# SID: 017382155
# Date: 10/01/2019
# 
# genetic_algorithm.py - this class implements a genetic_algorithm used to search
#						 the key space for the key that generates the most
#						 "English-like" key (or any language given user's corpus) 
##
##############################

class GeneticAlgorithm:
	"""
	Class that implements a Genetic Algorithm
	Searches key space to find the key that minimizes the loss function
	Loss function: -log(prob of ngram in corpus)

	...

	Attributes
	----------
	size : int
		size of population

	num_parents : int
		fraction of the population designated per epoch for mating

	size_extra : int
		size of extra individuals (chromosomes/keys) that will be 
		randomly initialized each epoch

	crossover_rate : float
		probability of crossover occuring

	mutation_rate : float
		probability of mutation occuring

	message : string
		original plaintext message

	text : list
		list of words in message including stopwords
	
	corpus : pandas.Dataframe
		corpus for decrypted message to be compared against

	monogram_stats : pandas.Dataframe
		monogram distribution for the ciphertext message

	corpus_<n>grams : pandas.Dataframe
		ngram distributions for the corpus
		** note that n is substituted by mono, bi, tri, quad, etc...**

	results : list
		contains best key for each epoch

	file_name : string
		name of file in which to save the results to

	sample_space : list
		list of all possible keys generated from monogram statistics of ciphertext

	population : list
		initial random sample of the sample (key) space, of size self.size

	scores : dictionary
		python dictionary containing all keys and the corresponding loss
	
	Methods
	-------
	getDistributions(self)
		returns the ngram Dataframes
	"""
	def __init__(self, 
				 message, 
				 corpus,
				 path,
				 file_name,
				 size=1e3,
				 frac_parents=0.75,
				 size_extra=150, 
				 crossover_rate=0.6,
				 mutation_rate=0.5):
		random.seed()

		self.size = int(size)
		self.num_parents = int(frac_parents * self.size)
		self.size_extra = size_extra
		self.crossover_rate = crossover_rate
		self.mutation_rate = mutation_rate

		self.message = message
		self.text = [word.lower() for word in message.split() if word.isalpha()]
		
		self.corpus = corpus
		self.monogram_stats = monogram_distribution(self.text)

		self.corpus_monograms, self.corpus_bigrams, self.corpus_trigrams, self.corpus_quadgrams = self.corpus.getDistributions()

		self.results = []
		self.file_name = path + file_name
		
		os.makedirs(path)
		self.result_file = open(self.file_name + ".txt", "w")

	def generate_population(self):
		'''
			Function to initialize the population
		'''
		monogram_keys = generate_keys(self.monogram_stats)
		self.sample_space = monogram_keys

		self.population = random.sample(self.sample_space, self.size)
		self.scores = {}

	def train(self, epochs=1e3, epsilon=1e-3):
		'''
			Function to run the genetic algorithm
			Plots and saves results of each epoch in text file

			Args:
				epochs: int, number of epochs to be run
				epsilon: float, tolerance for convergence
		'''
		epochs = int(epochs)

		self.generate_population()
		self.loss()

		for i in range(epochs):
			t_prev = time.time()

			self.select_pool()
			self.crossover()
			self.mutate()
			self.loss()

			self.result_file.write("------------------------\n")
			self.result_file.write("Epoch " + str(i) + "\n")
			self.result_file.write("Train time " + str(time.time() - t_prev) + "\n")
			self.result_file.write("Best key: " + self.best_key + "\n")
			self.result_file.write("Loss: " + str(self.scores[self.best_key]) + "\n")
			self.result_file.write(decrypt(self.message, self.best_key) + "\n\n")

			self.results.append(self.scores[self.best_key])


		plt.plot(list(range(epochs)), self.results)

		plt.xlabel('Epochs')
		plt.ylabel('Min Loss')
		plt.savefig(self.file_name + "png")

	def crossover(self):
		'''
			Crossover operator

			-Creates an offspring set (fixed size) from subset of the
				population (parents)
			-with probability self.crossover_rate, will take first half of key
				from parent 1 and second half from parent 2
				additionally another child will take first half of key
				from parent 2 and second half from parent 1
			-if an offspring has repeating characters, it is discarded and
				replaced with a random key from sample (key) space
				the random key is gauranteed to not be in 
				the parent set, current offpsring set, or the previous population
		'''
		self.offspring_size = self.size - self.num_parents - self.size_extra

		seen = []
		self.offspring = []
		for i in range(self.offspring_size):
			rand = random.uniform(0, 1)

			if rand < self.crossover_rate:
				par1 = random.sample(self.parents, 1)
				par2 = random.sample(self.parents, 1)

				while par1 in seen or par2 in seen:
					par1 = random.sample(self.parents, 1)
					par2 = random.sample(self.parents, 1)

				while par1 == par2:
					par2 = random.sample(self.parents, 1)

				seen.append(par1)
				seen.append(par2)

				par1 = list(par1)
				par2 = list(par2)

				crossover_point = int(len(par1)/2)

				offspring1 = par1[:crossover_point] + par2[crossover_point:]
				offspring2 = par2[:crossover_point] + par1[crossover_point:]

				junk1 = junk2 = False
				for x, count in Counter(offspring1).items():
					if count > 1:
						junk1 = True
						break

				for x, count in Counter(offspring2).items():
					if count > 1:
						junk1 = True
						break

				if not junk1:
					self.offspring.append(''.join(offspring1))

				if not junk2:
					self.offspring.append(''.join(offspring2))

		# get number of failed crossovers
		died = self.offspring_size - len(self.offspring)

		self.extra = []
		for i in range(self.size_extra + died):
			rand_key = random.sample(self.sample_space, 1)

			while rand_key in self.population or \
				  rand_key in self.parents or \
				  rand_key in self.offspring:
				rand_key = random.sample(self.sample_space, 1)

			self.extra += rand_key

		self.population = self.parents + self.offspring + self.extra

		random.shuffle(self.population)

	def loss(self):
		'''
			Function to calculate the loss for a given key

			1) decrypt the ciphertext message using each key
			2) get ngram statistics of the decrypted message
			3) for each ngram in the message check if that ngram is found in
			   corpus distribution, if it is not skip to step 5)
			4) add the loss of the ngram in the corpus to the loss function of
			   given key, skip to step 6)
			5) since the ngram was not in the corpus, penalize the loss function
			   with a large value
			6) get and save the best key

			Args:
				message: string, encrypted ciphertext
				key: string, list of alphabetical characters to substitute

			Returns:
				decrypted: string, the decrypted plaintext
		'''
		self.scores = {}
		for key in self.population:
			assert(len(key) == 26)
			decrypted = decrypt(self.message, key)
			decrypted = [word.lower() for word in decrypted.split() if word.isalpha()]
			quadgrams = quadgram_distribution(decrypted)
			#calc_entropy(quadgrams)
			calc_loss(quadgrams)

			for ngram, _ in quadgrams.iterrows():
				if not key in self.scores.keys():
						self.scores[key] = 0

				if self.corpus_quadgrams.index.isin([ngram]).any():
					index = self.corpus_quadgrams.index.get_loc(ngram)
					self.scores[key] += self.corpus_quadgrams.iloc[index]['loss']
				else:
					self.scores[key] += 9999999

		self.best_key = min(self.scores, key=self.scores.get)

	def mutate(self):
		'''
			Mutation operator

			With probability self.mutation_rate, swap two characters in a key
			Gauranteed to not mutate the best key
		'''
		new_pop = []

		for key in self.population:
			temp = ""
			rand = random.uniform(0, 1)

			if rand < self.mutation_rate and key != self.best_key:
				key = list(key)

				i = random.randint(0, 1e3) % len(key)
				j = random.randint(0, 1e3) % len(key)

				while i == j:
					j = random.randint(0, 1e3) % len(key)

				key[i], key[j] = key[j], key[i]

				key = ''.join(key)

			new_pop.append(key)

		self.population = new_pop

	def select_pool(self):
		'''
			Function to select mating pool (parents)

			Gets the n keys with the smallest losses
		'''
		self.parents = heapq.nsmallest(self.num_parents, self.scores)