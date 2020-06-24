# DLiA 2020 Mid Test - Nishit

## 1. Word representations: basic approaches (BoW, TF-iDF).

> To analyse text we need to represent as a vector. BoW and TF-iDF are methods through which we can represent a document as a single vector.
>
> For Bow we apply simple word embedding technique to create a matrix representation of our text where, each sentence(document) is tokenized into terms and we store the term frequency(TF) of each term/token in the docuent that appears in each document. The size of the matrix is `len(vocab)` x `len(text)`. It doesn't give us much information since more frequent terms have a higher count and they rarely give us more information about the text. TF-iDF solves this problem of noise. Instead of filling the BoW matrix with TF we multiply it with inverse of document frequency(iDF) which is how many documents contain the term.

## 2. Word embeddings (word2vec: linearity, skip-gram, negative sampling, key ideas)

> Linearity method takes the context of each word as its input and tries to predict the word corresponding to the context.
>
> Skip-gram takes each word(focus word) in the text and also takes one-by-one words that surround it within a defined window to feed to the NN which after trainin will predict the probability for each word to appear in the window around the focus word. This is computationally expensive.
>
> Negative sampling reduces the computation by implementing subsampling method where we eliminate a word from the text based on it's frequency. So, words like `the`, `a` which appear very often are not used for trainin reducing the number of weights we need to update.

## 3. Ways to work with text data (RNN, CNN, classical approaches)

> CNNs are deep, feed forward NNs where connections between nodes do not form a cycle. We detect patterns of word ngrams as the result of each convolution will fire when a special pattern is detected. By varying the size of the kernels and concatenating their outputs.
>
> In RNNs the connections between nodes form a directed graph along a sequence. RNN is a sequence of neural network blocks that are linked to each others like a chain.
>
> Classical approaches were very mathematical and statistics based as opposed to the ML algorithms which are more data intensive. It was more of a statistical modelling as opposed to statistical learning in ML algos. They take data and logic give an answer/output/label as opposed ot ML algos which take data and answer/output/label and give logic. Classical approaches were good for interpretation not prediction.

## 4. Attention mechanism, Self-attention mechanism

> A revolutionary improvement over encoder-decoder models, attention is one component of a network’s architecture, and is in charge of managing and quantifying the interdependence between the input and output elements (General Attention) and within the input elements (Self-Attention). A major improvement to tackle long-term dependency problem of RNNs and CNNs. It maps the important and relevant words from the input sentence and assign higher weights to these words, enhancing the accuracy of the output prediction. We calculate attention in three steps:
> 1. We take the query and each key and compute the similarity between the two to obtain a weight using dot product, splice, detector, etc.
> 2. Use softmax to normalize these weights
> 3. We weight these weights in conjunction with the corresponding values and obtain the final Attention.
>
> In self attention, each word in the sentence needs to undergo attention computation. The goal is to learn the dependencies between the words in the sentence and use that information to capture the internal structure of the sentence. This further improves the long-term learning capability of attention since self-attention is applied to both each word and all words together, no matter how distant they are, the longest possible path is one so that the system can capture long term dependency relationships.

## 5. Contextualized embeddings main idea.

> A word has different meaning based on it's context. The idea is to capture the different meanings of a word based on it's context. With contextualized embeddings this is what we do. So, in such a model the embedding of a word would have different vector representation in different contexts(documents). Transformer models capture these.

## 6. Transformer: encoder and decoder structure main details.

> A transformer is an encoder-decoder architecture model which uses attention mechanisms to forward a more complete picture of the whole sequence to the decoder at once rather than sequentially. he Encoder block has 1 layer of a Multi-Head Attention followed by another layer of Feed Forward Neural Network. The decoder, on the other hand, has an extra Masked Multi-Head Attention. The encoder and decoder blocks are multiple identical encoders and decoders stacked on top of each other. Both the encoder stack and the decoder stack have the same number of units. The number of encoder and decoder units is a hyperparameter.

## 7. BERT structure, main ideas (masking, pre-training on many problems)

> BERT is based on the Transformer model but, since BERT’s goal is to generate a language model, it only consists of the encoder mechanism.
>
> **Masking:** Before feeding word sequences into BERT, 15% of the words in each sequence are replaced with a `mask` token. The model then attempts to predict the original value of the masked words, based on the context provided by the other, non-masked, words in the sequence. The BERT loss function takes into consideration only the prediction of the masked values and ignores the prediction of the non-masked words.
>
> **Pre-training on many problems:** BERT can be used as a base to solve variety of specific language problems by adding a small layer to the core model. Classification problems can be solved by adding a classification layer on top of the Transformer output for the `CLS` token. A Q&A model can be trained by learning two extra vectors that mark the beginning and the end of the answer.  A NER model can be trained by feeding the output vector of each token into a classification layer that predicts the NER label.

## 8. Machine translation metrics, quality functions

> There are quite a few metrics for MT, BLEU, NIST, METEOR, TER. BLEU score compares a generated translation to one or more reference translations. It is quick and inexpensive to calculate, easy to understand, language independent, correlates highly with human evaluation and has been widely adopted. It is useful when directly comparing two different systems on the same set of test documents. Though BLEU scores hold no absolute meaning of quality, they simply provide a relative comparison of two or more outputs.
>
> TER or Translation Edit Rate assesses MT output by calculating the amount of changes a human translator would have to make in order for the MT output to match the reference sentence in meaning and fluency. TER is commonly used when MT output is produced for post-editing, as it can help to estimate the amount of work your translators will have to do to bring the MT up to human quality.