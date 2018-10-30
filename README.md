# Detecting-Homographic-Puns
A pun is a form of wordplay in which one signifier (e.g., a word or phrase) suggests two or more meanings. Puns where the two meanings share the same pronunciation are known as homophonic puns. Goal of the project is to detect these kind of puns.
# ECNU Algorithm
ECNU uses a supervised approach to pun detection. The authors collects a training set of 60 homographic and 60 heterographic puns, plus 60 proverbs and famous sayings, from various Web sources. 
The data is then used to train a classifier, using features derived from WordNet and word2vec embeddings. 
The ECNU pun locator is knowledge-based, determining each context word’s likelihood of being the pun on the basis of the distance between its sense vectors, or between its senses and the context.
