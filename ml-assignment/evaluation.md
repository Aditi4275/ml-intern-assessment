# Trigram Language Model

**N-gram counts storage**
- Used `defaultdict(int)` for storing unigram, bigram, and trigram counts.
- Separate dictionaries for each n-gram level to simplify probability computations and smoothing.

**Text cleaning and preprocessing**
- Converted all text to lowercase.
- Removed punctuation using `regex`.
- Tokenized by splitting on whitespace.

**Padding and vocabulary**
- Added two start tokens `(<s>)` and one end token `(</s>)` to sentences.
- Included padding tokens in the vocabulary.
- Built vocabulary from all unique tokens in training text.

**Handling unknown words and smoothing**
- Implemented `Laplace` (add-one) smoothing with fixed `alpha=1` to avoid zero probabilities.
- Used unigram, bigram, and trigram counts for smoothing fallback.
- Backed off from trigram to bigram to unigram probabilities when contexts were sparse or unseen.

**Text generation**
- Generation starts with `<s>, <s>` context.
- Calculates smoothed trigram probabilities to predict the next word.
- Backs off to bigram and unigram probabilities if trigram context is unseen.
- Normalizes probabilities and samples the next word using weighted random choice.
- Repeats until end token `</s>` or max length reached.

**Modularity and clarity**
- Encapsulated preprocessing, probability computations, and smoothing in separate methods.
- Clear method separation helps maintain and extend the model.

## Sample Input
```python
tale_of_two_cities.txt
```
## Sample Output
```python
Generated Text:
nightfall first tide plaintiff still secreted showering where beggars nor low possibleyou forlorner curtseys immature altered blest address frowning tilepaved definite healths steering organized unpromising shouldnt echoless misery suppression stealth bleeding wig english unglazed desolately healing mine pet begged gatheringplace highand its personages acknowledge sparks incommode ruin handwriting courtly carried
```