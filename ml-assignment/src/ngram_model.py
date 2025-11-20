import random
import re
from collections import defaultdict

class TrigramModel:
    def __init__(self):
        """
        Initializes the TrigramModel with empty data structures for n-gram counts. 
        """
        self.trigram_counts = defaultdict(int)
        self.bigram_counts = defaultdict(int)
        self.unigram_counts = defaultdict(int)
        self.vocab = set()
        self.is_fit = False
        self.total_unigrams = 0
        self.alpha = 1

    def pre_process(self, text):
        """
        Preprocesses the input text by converting to lowercase and removing punctuation.
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def fit(self, text):
        """
        Trains the trigram model on the given text.

        Args:
            text (str): The text to train the model on.
        """
        # 1. clean the text
        text = self.pre_process(text)

        # 2. tokenize the text
        words = text.split()

        # 3. padding the text
        tokens = ['<s>', '<s>'] + words + ['</s>']

        # Build vocabulary
        self.vocab = set(words) | {'<s>', '</s>'}
        
        # Initialize counts
        self.total_unigrams = sum(self.unigram_counts.values())

        # 4. count trigrams, bigrams and unigrams
        for i in range(len(tokens)):
            self.unigram_counts[tokens[i]] += 1
            if i < len(tokens) - 1:
                self.bigram_counts[(tokens[i], tokens[i+1])] += 1
            if i < len(tokens) - 2:
                self.trigram_counts[(tokens[i], tokens[i+1], tokens[i+2])] += 1
        
        self.total_unigrams = sum(self.unigram_counts.values())  
        self.is_fit = True

    def _trigram_prob(self, w1, w2, w3):
        """
        Returns the smoothed probability P(w3|w1,w2).
        """
        V = len(self.vocab)
        tri_count = self.trigram_counts.get((w1, w2, w3), 0)
        bi_count = self.bigram_counts.get((w1, w2), 0)
        # Laplace smoothing
        prob = (tri_count + self.alpha) / (bi_count + self.alpha * V)
        return prob
    
    def _bigram_prob(self, w2, w3):
        """
        Returns the smoothed bigram probability P(w3|w2).
        """
        V = len(self.vocab)
        bi_count = self.bigram_counts.get((w2, w3), 0)
        uni_count = self.unigram_counts.get(w2, 0)
        # Laplace smoothing
        prob = (bi_count + self.alpha) / (uni_count + self.alpha * V)
        return prob
    
    def _unigram_prob(self, w3):
        """
        Returns the smoothed unigram probability P(w3).
        """
        V = len(self.vocab)
        uni_count = self.unigram_counts.get(w3, 0)
        # Laplace smoothing
        prob = (uni_count + self.alpha) / (self.total_unigrams + self.alpha * V)
        return prob
    
    def generate(self, max_length=50):
        """
        Generates new text using the trained trigram model.

        Args:
            max_length (int): The maximum length of the generated text.

        Returns:
            str: The generated text.
        """
        if not self.is_fit:
            raise ValueError("Model must be fit before generating text.")
        
        # 1. Starting with the start tokens.
        w1, w2 = '<s>', '<s>'
        generated_words = []

        # 2. Probabilistically choosing the next word based on the current context.
        for _ in range(max_length):
            candidates = list(self.vocab)
            probs = []

            for w3 in candidates:
                # Try trigram probability first
                p = self._trigram_prob(w1, w2, w3)
                # If trigram context unseen, back off to bigram
                if self.bigram_counts.get((w1, w2), 0) == 0:
                    p = self._bigram_prob(w2, w3)
                # If bigram context also unseen, back off to unigram
                if self.unigram_counts.get(w2, 0) == 0:
                    p = self._unigram_prob(w3)
                probs.append(p)
        
            # Normalize probabilities
            total_prob = sum(probs)
            probs = [p / total_prob for p in probs]

            # Sample the next word based on the computed probabilities
            next_word = random.choices(candidates, weights=probs, k=1)[0]

            if next_word == '</s>':
                break
            
            generated_words.append(next_word)
            w1, w2 = w2, next_word

        return ' '.join(generated_words)
