import time
import genetic_algorithm as GA

from corpus import Corpus
from helper_functions import *
##############################
##
# Author: Matthew Buchholz
# SID: 017382155
# Date: 10/01/2019
# 
# cryptalyzer.py - driver program to run GA to break simple substitution cipher
##
##############################

##############################
##
# Texts used to build Corpus:
#  Alice’s Adventures in Wonderland, by Lewis Carroll
#  The Republic, by Plato
#  Siddhartha, by Herman Hesse
## 
##############################

texts = ['https://www.gutenberg.org/files/11/11-0.txt',
		 'http://www.gutenberg.org/cache/epub/1497/pg1497.txt',
		 'http://www.gutenberg.org/cache/epub/2500/pg2500.txt']


messages = ["fqjcb rwjwj vnjax bnkhj whxcq nawjv nfxdu mbvnu ujbbf nnc", 
			"oczmz vmzor jocdi bnojv dhvod igdaz admno ojbzo rcvot jprvi oviyv aozmo cvooj ziejt dojig toczr dnzno jahvi fdiyv xcdzq zoczn zxjiy", 
			"ejitp spawa qleji taiul rtwll rflrl laoat wsqqj atgac kthls iraoa twlpl qjatw jufrh lhuts qataq itats aittk stqfj cae", 
			"iyhqz ewqin azqej shayz niqbe aheum hnmnj jaqii yuexq ayqkn jbeuq iihed yzhni ifnun sayiz yudhe sqshu qesqa iluym qkque aqaqm oejjs hqzyu jdzqa diesh niznj jayzy uiqhq vayzq shsnj jejjz nshna hnmyt isnae sqfun dqzew qiead zevqi zhnjq shqze udqai jrmtq uishq ifnun siiqa suoij qqfni syyle iszhn bhmei squih nimnx hsead shqmr udquq uaqeu iisqe jshnj oihyy snaxs hqihe lsilu ymhni tyz"]

# initialize the corpus
corp = Corpus(texts, is_url=True)
# or alternatively, load the corpus from pickle files
#corp = Corpus(load=True)

for i, message in enumerate(messages):
	sec = time.time()

	model = GA.GeneticAlgorithm(message, corp, "./results/", "results" + str(i))

	model.train(epochs=1e3)

	print("Time " + str(i) + ": " + str((time.time() - sec)))