import urllib.request as url

from helper_functions import *
from spacy.lang.en.stop_words import STOP_WORDS
###############################################
##
# Author: Matthew Buchholz
# SID: 017382155
# Date: 10/01/2019
# 
# corpus.py - this class creates a Corpus from a given text source
##
###############################################

class Corpus:
	"""
	Class that represents a given Corpus
	Saves ngram statistics as pickle file

	...

	Attributes
	----------
	text : list
		list of strings containing each English word (stop words excluded)

	ngrams (one for each value of n) : pandas.Dataframe
		stores the ngrams and associated statistics:
		- relative frequency and negative log probabilities
	
	Methods
	-------
	getDistributions(self)
		returns the ngram Dataframes
	"""
	def __init__(self, texts=[], is_url=False, load=False):
		self.text = []

		if not load:
			if is_url:
				for site in texts:
					data = url.urlopen(site)
					for line in data:
						for word in str(line).split():
							if word.isalpha() and not word in STOP_WORDS:
								self.text.append(word.lower())

			self.monograms = monogram_distribution(self.text)
			self.bigrams   = bigram_distribution(self.text)
			self.trigrams  = trigram_distribution(self.text)
			self.quadgrams = quadgram_distribution(self.text)

			'''
			calc_entropy(self.monograms)
			calc_entropy(self.bigrams)
			calc_entropy(self.trigrams)
			calc_entropy(self.quadgrams)
			'''

			calc_loss(self.monograms)
			calc_loss(self.bigrams)
			calc_loss(self.trigrams)
			calc_loss(self.quadgrams)

			self.monograms.to_pickle("./data/monograms.pkl")
			self.bigrams.to_pickle("./data/bigrams.pkl")
			self.trigrams.to_pickle("./data/trigrams.pkl")
			self.quadgrams.to_pickle("./data/quadgrams.pkl")

		if load:
			self.monograms = pd.read_pickle("./data/monograms.pkl")
			self.bigrams   = pd.read_pickle("./data/bigrams.pkl")
			self.trigrams  = pd.read_pickle("./data/trigrams.pkl")
			self.quadgrams = pd.read_pickle("./data/quadgrams.pkl")

	def getDistributions(self):
		return (self.monograms, 
				self.bigrams, 
				self.trigrams,
				self.quadgrams)