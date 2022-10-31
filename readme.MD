### Assignment #4: Natural Language and Probabilistic Reasoning 

#### Due Mon Nov 21, start of class. 

The first portions of this assignment are written in a lab format, with a number of tasks to provide you with exposure to NLTK.

In order to complete this assignment, you will need [NLTK](https://www.nltk.org) (it’s in conda, if you use that).

We’ll be reading from the NLTK book, which can be found [here:](https://www.nltk.org/book/)

To submit: add code and documents to your GitHub repo for this assignment, along with a submission.py file that runs your code. 
There are several places where you're asked to measure or compare performance. Please add a document called HW4.pdf to your repo containing the answers to these questions.


Part 1. Getting warmed up. These tasks are intended to get you started with NLTK.

(5 points) Please read chapter 1, sections 1 and 3. Please feel free to skim the parts about Python as necessary. Choose one of the intro texts and compute its lexical diversity.

(5 points) Generate the bigrams for this text.

(5 points) Generate a FreqDist for this text to identify the 50 most common words. Filter the text to lower case, remove non-words and stopwords to see if you can generate a more informative distribution.

(5 points) Please read chapter 2, sections 1 and 2. Choose one of the Gutenberg texts and create a FreqDist with it as you did above. Be sure to remove stop words, convert to lower case and remove non-words.

(5 points) Next, create a ConditionalFreqDist (CFD), where the conditions are the fileids in the Gutenberg corpus and the events are the words in Gutenberg.words(), filtered as above to convert to lower case, and remove stop words and non-words.

Part 2: Document clustering. (25 points)

In this part, you'll implement the k-means algorithm and use it to classify documents.

We'll use the Brown corpus for this task.

To begin, select at least 5 categories from the Brown corpus.

Using the FreqDist object, construct a word vector for each document (strip punctuation, convert to lowercase, and remove stopwords). Write a function called k_means that takes as input a list of FreqDist objects and a number (k) of clusters and returns a list of five lists, one for each cluster, containing the documents in each cluster, using Euclidean distance as your metric.

Compute your performance. What fraction of the articles were placed in the proper clusters? To measure this, for each cluster determine the most common category, and then count the fraction of articles in that category.

Compare your performance to that of the [nltk.cluster](https://www.nltk.org/api/nltk.cluster.html) package.

Part 3: Supervised learning with Naive Bayes. (25 points)

Clustering is an unsupervised algorithm. It constructs a hypothesis using only the relationships within the data.

On the other hand, Naive Bayes is a supervised algorithm. It uses labeled training data to try to make generalizations.

Using the same five categories as above, construct a Naive Bayes classifier. You will probably want to re-use your FreqDist vectors from before.

Your program should: Pre-process all the data for each category as above - remove stopwords, punctuation, and convert to lower case.

For each category, construct a FreqDist. You'll use this to compute P(word | category).

Then, implement a function called classify. It should take as input a vector representing an unknown document, and our FreqDists for each category. (Please feel free to use a ConditionalFreqDist here to make things easier.) It should then compute P(category | word1, word2, ..., wordn) for each category, and return the most likely category.

To test your performance, use five-fold cross-validation. You're welcome to re-use your code from assignment 3, or you can use [sklearn's implementation](https://scikit-learn.org/stable/modules/cross_validation.html)

Compare your algorithm's performance to the [Naive Bayes classifier included with NLTK](https://www.nltk.org/api/nltk.classify.naivebayes.html?highlight=naive%20bayes#module-nltk.classify.naivebayes).

Part 4: Hidden Markov Models (25 points)

(Note: this is derived from an assignment in AAAI's Model Assignments workshop)

In this assignment you'll be implementing two algorithms associated with Hidden Markov Models.

You'll be building off of the code presented in HMM.py. There's also some included data to use.

The first set of files are .trans files. They contain the transition probabilities. two_english models the transition between 'C' for Consonant and 'V' for Vowel in English. partofspeech.browntags.trained models transitions between parts of speech in the Brown corpus.

The second set of files are .emit files. These contain the probability of emitting a particular output from that state. These are learned from data, and so contain errors (especially two_english).

(5 points). Use the included code to implement load. Use two_english as a sample file to work with. You should be able to do:

    model = HMM()
    model.load('two_english')

You should store the transitions and emissions as dictionaries of dictionaries. e.g. {'#': {'C': 0.814506898514, 'V': 0.185493101486}, 'C': {'C': 0.625840873591, 'V': 0.374159126409}, 'V': {'C': 0.603126993184, 'V': 0.396873006816}}

(10 points) Implement generate. It should take an integer n, and return a random observation of length n. To generate this, start in the initial state and repeatedly select successor states at random, using the probability as a weight, and then select an emission, again using the probability as a weight. You may find either numpy.random.choice or random.choices very helpful here.

You should be able to run it with the pre-trained probabilities for the Brown corpus, like so:

python hmm.py partofspeech.browntags.trained --generate 20

which generates 20 random observations.

Here are two sample observations:

DET ADJ . ADV ADJ VERB ADP DET ADJ NOUN VERB ADJ NOUN
the semi-catatonic , quite several must of an western bridge cannot spectacular analyses
DET NOUN ADP NOUN CONJ DET VERB DET NOUN NOUN NOUN ADP DET NOUN
whose light for wall and the learned the hull postmaster trash in his peters


(15 points) Next, implement Viterbi. This tells us, for a sequence of observations, the most likely sequence of states. You should be able to run this like so:

python hmm.py partofspeech.browntags.trained --viterbi ambiguous_sents.obs

This uses the HMM parameters in partofspeech.browntags.trained.{trans,emit} to compute the best sequence of part-of-speech tags for each sentence in ambiguous_sents.obs, and writes it to ambiguous_sents.out.

You might find it helpful to use a numpy array to hold the matrix.

Compare the output file to ambiguous_sents.tagged.obs.

(grad students only) Please read [this article](https://www.theatlantic.com/magazine/archive/2013/11/the-man-who-would-teach-machines-to-think/309529/) 
about Douglas Hofstadter, which also serves as a nice summary of the history of AI and the debates over the value of developing machines that think like humans. (As an aside: If you have not read Hofstadter's book [Godel, Escher, Bach](https://en.wikipedia.org/wiki/G%C3%B6del,_Escher,_Bach), I strongly recommend it.)

Prepare a summary or critique of this article that addresses the following questions:

- Hofstadter is particularly interested in understanding the way humans think. What sorts of reasoning mechanisms does he study?
    
- The article includes a quote from our text: “The quest for ‘artificial flight’ succeeded when the Wright brothers and others stopped imitating birds and started … learning about aerodynamics,” What does this mean? Why is it relevant to AI?

- What was Candide? Why did it change the way we thought about machine translation?

- The article also contains a quote from the last chapter of AIMA: perhaps AI has become too much like the man who tries to get to the moon by climbing a tree: “One can report steady progress, all the way to the top of the tree.” What does this mean? How does it relate to Candide and the ways in which big data and machine learning have changed AI?

 