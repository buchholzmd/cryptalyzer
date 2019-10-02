import numpy as np
import pandas as pd
import itertools as it
import matplotlib.pyplot as plt

from collections import Counter
###############################
##
# Author: Matthew Buchholz
# SID: 017382155
# Date: 10/01/2019
# 
# helper_functions.py - this class implements various helper functions
#	for the corpus, cryptanalyzer, and genetic_algorithm programs
##
###############################

def encrypt(message, key):
	'''
		Function to decrypt a given piece of text given some cryptographic key
		Using simple substitution

		Args:
			message: string, encrypted ciphertext
			key: string, list of alphabetical characters to substitute

		Returns:
			decrypted: string, the decrypted plaintext
	'''
	encrypted = ""
	# letter frequencies acquired from letterfrequency.org
	#alphabet = list('etaoinsrhldcumfpgwybvkxjqz')
	alphabet = list('abcdefghijklmnopqrstuvwxyz')
	for letter in message:
		if(letter == " "):
				encrypted += letter
		else:		
			for i, char in enumerate(key):
				
				if(letter == alphabet[i]):
					encrypted += char

	return encrypted

def decrypt(message, key):
	'''
		Function to decrypt a given piece of text given some cryptographic key
		Using simple substitution

		Args:
			message: string, encrypted ciphertext
			key: string, list of alphabetical characters to substitute

		Returns:
			decrypted: string, the decrypted plaintext
	'''
	decrypted = ""
	# letter frequencies acquired from letterfrequency.org
	#alphabet = list('etaoinsrhldcumfpgwybvkxjqz')
	alphabet = list('abcdefghijklmnopqrstuvwxyz')
	for letter in message:
		if(letter == " "):
				decrypted += letter
		else:		
			for i, char in enumerate(key):
				
				if(letter == char):
					decrypted += alphabet[i]

	return decrypted

def generate_keys(distribution):
	'''
		Function to generate cryptographic keys given ngrams statistics
		-Generates a key by substituting for the most frequent ngrams
		-If two characters(ngrams) are equiprobable, get all permutations
			of the two characters(ngrams) and generate all possible keys from
			the given permutations

		Args:
			distribution: pandas.Dataframe, ngrams with relative frequencies

		Returns:
			keys: list, list of cryptographic keys
	'''
	keys = []
	temp = []
	dist_map = []
	seen = ""
	first = True
	
	iterator = distribution.iterrows()

	prev_prob = 0
	for letter, probs in iterator:

		# if next ngram does not have the same probability as previous ngram
		# 
		# the flag first makes sure that you do not miss the first ngram
		if(abs(prev_prob - probs['rel_freq']) >= 1e-5):
			if not first:
				dist_map.append(''.join(temp))
				temp = []

			first = False
			seen += letter
			temp.append(letter)
			prev_prob = probs['rel_freq']
			
			letter, probs = next(iterator)

		# temp stores a list of ngrams of the same frequency
		# this is used to later get all permutations of 
		# 	equally probable substrings
		while(abs(prev_prob - probs['rel_freq']) < 1e-5):
			seen += letter
			temp.append(letter)
			prev_prob = probs['rel_freq']
			try:
				letter, probs = next(iterator)
			except StopIteration:
				break

		# create a string from equally probable ngrams
		dist_map.append(''.join(temp))

		# reinitialize temp and run the initial check again
		# this is done because the python iterator will 
		# 	skip this next ngram otherwise
		temp = []
		if(abs(prev_prob - probs['rel_freq']) > 1e-5):
			seen += letter
			temp.append(letter)
			prev_prob = probs['rel_freq']

	# if temp has leftover ngrams, create a string and append to the list of
	# concatenated equiprobable ngrams
	if(not len(temp) == 0):
		dist_map.append(''.join(temp))

	# get all permutations of equiprobable ngrams
	perm_map = []
	for substring in dist_map:
		perm_map.append(permutation(substring))

	# make sure that the whole alphabet is accounted for
	extra = ""
	for letter in 'abcdefghijklmnopqrstuvwxyz':
		if not letter in seen:
			extra += letter

	# get permutations of the extra characters in alphabet
	# since permutation is O(n!), split up the strings into thirds if n > 5
	if len(extra) > 5:
		substring1 = extra[:int(len(extra)/3)]
		substring2 = extra[int(len(extra)/3):int(2*len(extra)/3)]
		substring3 = extra[int(2*len(extra)/3):]

		perms1 = permutation(substring1)
		perms2 = permutation(substring2)
		perms3 = permutation(substring3)

		extra_strings = [ss1 + ss2 + ss3 for ss3 in perms3 for ss2 in perms2 for ss1 in perms1]

	else:
		extra_strings = permutation(extra)

	# append the extra permutations and finally create the list of keys
	perm_map.append(extra_strings)
	keys = [''.join(x) for x in list(it.product(*perm_map))]

	return keys

def calc_entropy(distribution):
	'''
		Function to calculate the entropy of each ngram, add as new column in
			the Dataframe

		Args:
			distribution: pandas.Dataframe, ngrams with relative frequencies
	'''
	probs = distribution['rel_freq']
	distribution['entropy'] = -(probs*np.log(probs))

def calc_loss(distribution):
	'''
		Function to calculate the loss of each ngram (used for GA)
			added as new column in the Dataframe

		loss(ngram) = - log(prob of ngram)

		Args:
			distribution: pandas.Dataframe, ngrams with relative frequencies
	'''
	probs = distribution['rel_freq']
	#distribution['loss'] = (1/probs)*(np.log((1/probs)))
	distribution['loss'] = -np.log(probs)

def monogram_distribution(corpus, i=0, save_fig=False):
	'''
		Function to generate the distibution of monograms (characters) in a
			given piece of text

		Args:
			corpus: list, list of words in corpus
			i: int, index of message for saving the distribution plot
			save_fig: bool, flag whether one wants to plot and save 
							the distribution plot 

		Returns:
			df: pandas.Dataframe, dataframe containing monograms and relative freqs
	'''
	chars = [char for word in corpus for char in word]
	freq_count = Counter(chars)

	df = pd.DataFrame.from_dict(freq_count, orient='index', columns=['count'])
	df = df.sort_values('count', ascending=False)
	df['count'] /= df['count'].sum()
	df = df.rename(columns = {'count': 'rel_freq'})

	if save_fig:
		fig = df.plot(kind='bar', title=('Monogram Distribution ' + str(i)), sort_columns=True).get_figure()
		
		plt.xlabel('Monograms')
		plt.ylabel('Relative Frequency')
		fig.savefig('./monograms_' + str(i) + '.png')

	return df

def bigram_distribution(corpus, i=0, save_fig=False):
	'''
		Function to generate the distibution of bigrams in a
			given piece of text

		Args:
			corpus: list, list of words in corpus
			i: int, index of message for saving the distribution plot
			save_fig: bool, flag whether one wants to plot and save 
							the distribution plot 

		Returns:
			df: pandas.Dataframe, dataframe containing bigrams and relative freqs
	'''
	bigrams = []
	for word in corpus:
		for j in range(len(word)-1):
			bigrams.append(word[j] + word[j+1])

	freq_count = Counter(bigrams)

	df = pd.DataFrame.from_dict(freq_count, orient='index', columns=['count'])
	df = df.sort_values('count', ascending=False)
	df['count'] /= df['count'].sum()
	df = df.rename(columns = {'count': 'rel_freq'})

	if save_fig:
		fig = df.plot(kind='bar', title=('Bigram Distribution ' + str(i)), sort_columns=True).get_figure()
		
		plt.xlabel('Bigrams')
		plt.ylabel('Relative Frequency')
		fig.savefig('./bigrams_' + str(i) + '.png')

	return df

def trigram_distribution(corpus, i=0, save_fig=False):
	'''
		Function to generate the distibution of trigrams in a
			given piece of text

		Args:
			corpus: list, list of words in corpus
			i: int, index of message for saving the distribution plot
			save_fig: bool, flag whether one wants to plot and save 
							the distribution plot 

		Returns:
			df: pandas.Dataframe, dataframe containing trigrams and relative freqs
	'''
	trigrams = []
	for word in corpus:
		for j in range(len(word)-2):
			trigrams.append(word[j] + word[j+1] + word[j+2])

	freq_count = Counter(trigrams)

	df = pd.DataFrame.from_dict(freq_count, orient='index', columns=['count'])
	df = df.sort_values('count', ascending=False)
	df['count'] /= df['count'].sum()
	df = df.rename(columns = {'count': 'rel_freq'})

	if save_fig:
		fig = df.plot(kind='bar', title=('Trigram Distribution ' + str(i)), sort_columns=True).get_figure()
		
		plt.xlabel('Trigrams')
		plt.ylabel('Relative Frequency')
		fig.savefig('./trigrams_' + str(i) + '.png')

	return df

def quadgram_distribution(corpus, i=0, save_fig=False):
	'''
		Function to generate the distibution of quadgrams in a
			given piece of text

		Args:
			corpus: list, list of words in corpus
			i: int, index of message for saving the distribution plot
			save_fig: bool, flag whether one wants to plot and save 
							the distribution plot 

		Returns:
			df: pandas.Dataframe, dataframe containing quadgrams and relative freqs
	'''
	quadgrams = []
	for word in corpus:
		for j in range(len(word)-3):
			quadgrams.append(word[j] + word[j+1] + word[j+2] + word[j+3])

	freq_count = Counter(quadgrams)

	df = pd.DataFrame.from_dict(freq_count, orient='index', columns=['count'])
	df = df.sort_values('count', ascending=False)
	df['count'] /= df['count'].sum()
	df = df.rename(columns = {'count': 'rel_freq'})

	if save_fig:
		fig = df.plot(kind='bar', title=('Quadgram Distribution ' + str(i)), sort_columns=True).get_figure()
		
		plt.xlabel('Quadgrams')
		plt.ylabel('Relative Frequency')
		fig.savefig('./quadgrams_' + str(i) + '.png')

	return df

def perm_helper(word, perm):
	'''
		Helper function for permutation
		Utilizes backtracking to traverse the recursion tree

		Args:
			word: list, list of characters in string
			perm: list, the current (sub)permutation generated by the recursion

		Returns:
			perms: list, list of lists containing all permutations of characters
	'''
	perms = []

	if len(word) == 0:
		return ([perm + []] + [])

	for i in range(len(word)):
		perm.append(word[i])
		perms.extend(perm_helper(word[0:i] + word[i+1:], perm))
		perm.remove(word[i])

	return perms

def permutation(word):
	'''
		Helper function for permutation

		Args:
			word: string, string to generate permutations of

		Returns:
			perms: list, list of strings containing all permutations
	'''
	perms = []
	for x in perm_helper(list(word), []):
		perms.append(''.join(x))
	return perms