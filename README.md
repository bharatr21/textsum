# textsum
Exploring Text Summarization Techniques

[Introduction](#introduction)
- [Abstractive Summarization](#abstractive-summarization)
- [Extractive Summarization](#extractive-summarization)
- [Metrics](#metrics)

[Baselines](#baselines)
- [Centroid](#centroid)
- [Luhn](#luhn)
- [Latent Semantic Analysis](#latent-semantic-analysis)
- [TextRank](#textrank)
- [LexRank](#lexrank)


[BigramRank](#bigramrank)
- [Ends](#ends)
- [Middle](#middle)

[Results](#results)

[Deep Learning](#deep-learning)

## Introduction
**Text Summarization** is the task of generating a short and concise summary that captures the main ideas of the source text. There are two main approaches to text summarization:
### Abstractive Summarization
Abstractive methods build an internal semantic representation of the original content, and then use this representation to create a summary that is closer to what a human might express.
The generated summaries potentially contain new phrases and sentences that may not appear in the source text.\
The current state of the art can be tracked [here](https://paperswithcode.com/task/abstractive-text-summarization)

### Extractive Summarization
Extractive methods use the original text to create a summary that is as close as possible to the original content. Here, content is extracted from the original data, but the extracted content is not modified in any way. 
Most of the techniques explored in this project are related to extractive summarization.\
The current state of the art can be tracked [here](https://paperswithcode.com/task/extractive-document-summarization)

### Metrics
The main set of metrics used to evaluate the performance of text summarization techniques are [ROUGE](https://en.wikipedia.org/wiki/ROUGE_(metric)) (Recall-Oriented Understudy for Gisting Evaluation) of which we use the following:

<!-- $ROUGE-N = \frac{\sum_{S \in \{Reference\ Summaries\}}\sum_{gram_n \in S}{Count_{match}(gram_n)}}{\sum_{S \in \{Reference\ Summaries\}}\sum_{gram_n \in S}{Count(gram_n)}}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?ROUGE-N%20%3D%20%5Cfrac%7B%5Csum_%7BS%20%5Cin%20%5C%7BReference%5C%20Summaries%5C%7D%7D%5Csum_%7Bgram_n%20%5Cin%20S%7D%7BCount_%7Bmatch%7D(gram_n)%7D%7D%7B%5Csum_%7BS%20%5Cin%20%5C%7BReference%5C%20Summaries%5C%7D%7D%5Csum_%7Bgram_n%20%5Cin%20S%7D%7BCount(gram_n)%7D%7D"> 

where  <!-- $gram_n$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?gram_n"> stands for the <!-- $n-grams$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?n-grams"> appearing in <!-- $S$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?S"> or the reference summaries.\
We use the `sumeval` package which implements this method.

An  alternative implementation of the metric which uses precision, recall and F-measures provided by the `rouge` package in Python is as follows:

Precision <!-- $P_n = \frac{Count_{overlap}{(n-grams\ in\ Candidate\  and \ Reference\ Summaries)}}{Count{(n-grams\ in\ Candidate\ Summary)}}$ --> <img style="transform: translateY(0.3em); background: white;" src="https://latex.codecogs.com/svg.latex?P_n%20%3D%20%5Cfrac%7BCount_%7Boverlap%7D%7B(n-grams%5C%20in%5C%20Candidate%5C%20%20and%20%5C%20Reference%5C%20Summaries)%7D%7D%7BCount%7B(n-grams%5C%20in%5C%20Candidate%5C%20Summary)%7D%7D">

Recall <!-- $R_n  = \frac{Count_{overlap}{(n-grams\ in\ Candidate\  and \  Reference\ Summaries)}}{Count{(n-grams\ in\ Reference\ Summary)}}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?R_n%20%20%3D%20%5Cfrac%7BCount_%7Boverlap%7D%7B(n-grams%5C%20in%5C%20Candidate%5C%20%20and%20%5C%20%20Reference%5C%20Summaries)%7D%7D%7BCount%7B(n-grams%5C%20in%5C%20Reference%5C%20Summary)%7D%7D">

F-score <!-- $F = \frac{(1 + \beta^2) P_n R_n}{\beta^2 P_n + R_n}$ --> 
<img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?F%20%3D%20%5Cfrac%7B(1%20%2B%20%5Cbeta%5E2)%20P_n%20R_n%7D%7B%5Cbeta%5E2%20P_n%20%2B%20R_n%7D">

Another metric used is finding the longest common subsequence (LCS) between a candidate summary and the original reference summary called <img src="https://latex.codecogs.com/svg.image?ROUGE-L" title="ROUGE-L"/>. Let us consider <img src="https://latex.codecogs.com/svg.image?X" title="X" /> as a reference summary of length <img src="https://latex.codecogs.com/svg.image?m" title="m" /> and <img src="https://latex.codecogs.com/svg.image?Y" title="Y" /> as a candidate summary of length <img src="https://latex.codecogs.com/svg.image?n" title="n" />. Here, both the `rouge` and `sumeval` packages use the same implementation as follows:

Precision <!-- $P_{lcs} = \frac{LCS(X,Y)}{n}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?P_%7Blcs%7D%20%3D%20%5Cfrac%7BLCS(X%2CY)%7D%7Bn%7D">

Recall <!-- $R_{lcs} = \frac{LCS(X,Y)}{m}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?R_%7Blcs%7D%20%3D%20%5Cfrac%7BLCS(X%2CY)%7D%7Bm%7D"> and

F-score <!-- $F_{lcs} = \frac{(1 + \beta^2) P_{lcs}R_{lcs}}{\beta^2P_{lcs} + R_{lcs}}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?F_%7Blcs%7D%20%3D%20%5Cfrac%7B(1%20%2B%20%5Cbeta%5E2)%20P_%7Blcs%7DR_%7Blcs%7D%7D%7B%5Cbeta%5E2P_%7Blcs%7D%20%2B%20R_%7Blcs%7D%7D">

where <!-- $LCS(X,Y)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?LCS(X%2CY)"> is the length of the longest common subsequence between <!-- $X$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?X"> and <!-- $Y$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?Y">.

## Baselines
**NOTE:** A lot of the baselines explained here are implemented with the `sumy` package in Python.
### Centroid
This method is a variation of [TF-IDF](https://en.wikipedia.org/wiki/Tf-idf) where the **centroid** is the average of the TF-IDF scores of all the words in the text. 
The simple version of the TF-IDF formula is as follows:
For term <!-- $i$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?i"> in document <!-- $j$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?j">,
<!-- $w_{i,j} = tf_{i,j} \times \log (\frac{N}{df_{i}})$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?w_%7Bi%2Cj%7D%20%3D%20tf_%7Bi%2Cj%7D%20%5Ctimes%20%5Clog%20(%5Cfrac%7BN%7D%7Bdf_%7Bi%7D%7D)"> 
where <!-- $tf_{i,j} = $ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?tf_%7Bi%2Cj%7D%20%3D"> number of occurrences of term <!-- $i$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?i"> in <!-- $j$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?j">

<img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?df_%7Bi%2Cj%7D%20%3D"> the number of documents that contain <!-- $i$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?i">

<img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?N%20%3D"> the total number of documents in the corpus.

### Luhn
This method can be extended to use abstracts/excerpts instead of complete sentences and works as follows:
- First identify the sentence which consists of the cluster containing the maximum number of **significant** words. (This can be determined by another frequency based algorithm such as TF-IDF or thresholding)
- Significance of Sentence = <!-- $\frac{[Count( significant\ words\ in\ sentence)]^2}{Count(words\ in\ sentence)}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?%5Cfrac%7B%5BCount(%20significant%5C%20words%5C%20in%5C%20sentence)%5D%5E2%7D%7BCount(words%5C%20in%5C%20sentence)%7D">
![Luhn](assets/Luhn.png)

### Latent Semantic Analysis
Latent Semantic Analysis (LSA) applies the concept of **S**ingular **V**alue **D**ecomposition (SVD) to text summarization.
A term by sentence matrix <!-- $A = [A_1, A_2, ..., A_n]$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?A%20%3D%20%5BA_1%2C%20A_2%2C%20...%2C%20A_n%5D"> is created with <!-- $A_k$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?A_k">  representing the weighted term frequency of sentence <!-- $k$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?k"> in the document. In a document where there are <!-- $m$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?m"> terms and <!-- $n$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?n"> sentences, <!-- $A$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?A"> will be an <!-- $m \times n$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?m%20%5Ctimes%20n"> matrix.
Given an <!-- $m \times n$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?m%20%5Ctimes%20n"> matrix <!-- $A$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?A">, where <!-- $m \geq n$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?m%20%5Cgeq%20n">, the SVD of <!-- $A$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?A"> is defined as: <!-- $U \sum V^T$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?U%20%5Csum%20V%5ET">

where <!-- $V$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?V"> is an <!-- $n \times n$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?n%20%5Ctimes%20n"> orthogonal matrix,

<img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?U"> is an <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?m%20%5Ctimes%20n"> column-orthonormal matrix with left-singular vectors,

Σ is an <!-- $n \times n$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?n%20%5Ctimes%20n"> diagonal matrix, whose diagonal elements are non-negative singular values in descending order.

![LSA](assets/LSA.png)
For each sentence vector in matrix <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?V"> (its components are multiplied by corresponding singular values) we compute its length. The reason of the multiplication is to favour the index values in the matrix <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?V"> that correspond to the highest singular values (the most significant topics). Formally:
<!-- $s_k = \sqrt{\sum_{i=1}^{n}v_{k,i}^2\sigma_i^2}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?s_k%20%3D%20%5Csqrt%7B%5Csum_%7Bi%3D1%7D%5E%7Bn%7Dv_%7Bk%2Ci%7D%5E2%5Csigma_i%5E2%7D">
where <!-- $s_k$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?s_k"> is the length of the vector representing the <!-- $k^{th}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?k%5E%7Bth%7D"> sentence in the modified latent vector space.
### TextRank
TextRank is a **graph-based** model for many NLP applications, where the importance of a single sentence depends on the importance of the entire document, and with respect to other sentences in a mutually recursive fashion, similar to [HITS](https://en.wikipedia.org/wiki/HITS_algorithm) or [PageRank](https://en.wikipedia.org/wiki/PageRank).
The score of a “text unit” or vertex $V$ in a graph (here vertex $V$ could represent any lexical unit such as a token, word, phrase, or a sentence) is calculated as follows:
<!-- $WS(V_i) = (1 - d) + d \times \sum_{V_j \in In(V_i)}\frac{w_{ji}}{\sum_{V_k \in Out(V_j)}w_{jk}}WS(V_j)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?WS(V_i)%20%3D%20(1%20-%20d)%20%2B%20d%20%5Ctimes%20%5Csum_%7BV_j%20%5Cin%20In(V_i)%7D%5Cfrac%7Bw_%7Bji%7D%7D%7B%5Csum_%7BV_k%20%5Cin%20Out(V_j)%7Dw_%7Bjk%7D%7DWS(V_j)">

For the task of sentence (summary) extraction, the goal is to rank entire sentences, and therefore a vertex is added to the graph for each sentence in the text. Two sentences are connected if there is a “similarity” relation between them, where “similarity” is measured as a function of their content overlap (common tokens). For two sentences <!-- $S_i$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?S_i"> and <!-- $S_j$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?S_j"> the similarity between them is:

<!-- $Sim(S_i, S_j) = \frac{|\{w_k | w_k \in S_i \&\ w_k \in S_j \}|}{\log(|S_i|) + \log(|S_j|)}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?Sim(S_i%2C%20S_j)%20%3D%20%5Cfrac%7B%7C%5C%7Bw_k%20%7C%20w_k%20%5Cin%20S_i%20%5C%26%5C%20w_k%20%5Cin%20S_j%20%5C%7D%7C%7D%7B%5Clog(%7CS_i%7C)%20%2B%20%5Clog(%7CS_j%7C)%7D"> 
where  <!-- $w_k$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?w_k">  is a token / "text unit" in the sentences.

TextRank takes into account both the local context of a text unit (vertex) and the information recursively drawn from the entire text document (graph).

This could be enhanced  by considering the text units as words and then giving an improved similarity score to the sentences by considering PageRank scores of tokens between <!-- $S_i$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?S_i"> and <!-- $S_j$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?S_j">
### LexRank
Another graph-based model for many NLP applications, this method performs **random walks** or **random traversals** through the graph, where the lexical unit or vertex <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?V"> is a sentence, this time it uses the *centrality* of each sentence in a document (cluster) to rank the sentences.

Here, the  similarity score between two sentences <!-- $x$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?x"> and <!-- $y$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?y"> is given as:
<!-- $idf-modified-cosine(x,y) = \frac{\sum_{w \in (x, y)}tf_{w,x}tf_{w,y}(idf_w)^2}{\sqrt{\sum_{x_i \in x}(tf_{x_i,x}idf_{x_i})^2} \times {\sqrt{\sum_{y_i \in y}(tf_{y_i,y}idf_{y_i})^2}}}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?idf-modified-cosine(x%2Cy)%20%3D%20%5Cfrac%7B%5Csum_%7Bw%20%5Cin%20(x%2C%20y)%7Dtf_%7Bw%2Cx%7Dtf_%7Bw%2Cy%7D(idf_w)%5E2%7D%7B%5Csqrt%7B%5Csum_%7Bx_i%20%5Cin%20x%7D(tf_%7Bx_i%2Cx%7Didf_%7Bx_i%7D)%5E2%7D%20%5Ctimes%20%7B%5Csqrt%7B%5Csum_%7By_i%20%5Cin%20y%7D(tf_%7By_i%2Cy%7Didf_%7By_i%7D)%5E2%7D%7D%7D">

where <!-- $tf_{w,s}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?tf_%7Bw%2Cs%7D"> is the number of occurrences of word <!-- $w$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?w"> in sentence <!-- $s$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?s">, and <!-- $idf_{w}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?idf_%7Bw%7D"> is the inverse document frequency of word <!-- $w$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?w">.

![LexRank](assets/LexRank.png)

This is an example of a weighted-cosine similarity graph generated for a cluster where <!-- $d_is_j$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?d_is_j"> represents document i, sentence j.

### BigramRank
We proposed a new algorithm BigramRank as a new **graph based model** which considers the basic lexical unit as bigrams (two consecutive tokens/words). This considers the frequency of bigrams (two consecutive words) and assigns the sentence relevance score for a sentence <!-- $S$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?S"> as:
Sentence Relevance Score of S is given as:
 <!-- $Score(S) = \frac{\sum_{Bigrams \in S}Bigrams}{\sum_{Bigrams \in Document}Bigrams}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?Score(S)%20%3D%20%5Cfrac%7B%5Csum_%7BBigrams%20%5Cin%20S%7DBigrams%7D%7B%5Csum_%7BBigrams%20%5Cin%20Document%7DBigrams%7D">
 
![BigramRank](assets/BigramRank.png)

This is the graph generated for the sentences “Windows 7 is a bit faster. Windows 7 is a little faster.” The bigram (Windows, 7) occurs 2 times, (little, faster) occurs 1 time.
#### Ends
The bigrams at the beginning and end of the sentence are considered slightly more important than bigrams occurring at the middle of a sentence, under the assumption that human readers may just **skim through the beginning and end of a large sentence** to look for the meaning of what the sentence conveys. To do this, the existing bigram frequency weights are multiplied elementwise by a **kernel**, which is <!-- $[1...0...1]$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?%5B1...0...1%5D"> which has weights decreasing from <!-- $[1, 0]$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?%5B1%2C%200%5D"> gradually in steps of <!-- $\frac{2}{(length\ of\ sentence-1)}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?%5Cfrac%7B2%7D%7B(length%5C%20of%5C%20sentence-1)%7D">  (where the weight is 0 at the exact middle bigram of the sentence, which means the middle/median bigram is not relevant to the sentence) and then increases from <!-- $[0, 1]$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?%5B0%2C%201%5D">, again in steps of  <!-- $\frac{2}{(length\ of\ sentence-1)}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?%5Cfrac%7B2%7D%7B(length%5C%20of%5C%20sentence-1)%7D">
Eg: - Kernel of length 5 is <!-- $[1,0.5,0,0.5,1]$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?%5B1%2C0.5%2C0%2C0.5%2C1%5D">, length 4 is <!-- $[1,0,0,1]$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?%5B1%2C0%2C0%2C1%5D">, length 9 is <!-- $[1,0.75,0.5,0.25,0,0.25,0.5,0.75,1]$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?%5B1%2C0.75%2C0.5%2C0.25%2C0%2C0.25%2C0.5%2C0.75%2C1%5D">
#### Middle
This assumes the opposite, that is, when humans read a sentence, they may tend to **skip to the middle part of a sentence to get information and the beginning is full of transition words/conjunctions.** In this case, the rest of the algorithm remains same and the **kernel gets flipped**, that is, first an increase from <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?%5B0%2C%201%5D"> in steps of <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?%5Cfrac%7B2%7D%7B(length%5C%20of%5C%20sentence-1)%7D"> and then a decrease from <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?%5B1%2C%200%5D"> with the same step.

Eg: - Kernel of length 5 is <!-- $[0,0.5,1,0.5,0]$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?%5B0%2C0.5%2C1%2C0.5%2C0%5D">, length 4 is <!-- $[0,1,1,0]$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?%5B0%2C1%2C1%2C0%5D">, length 9 is <!-- $[0,0.25,0.5,0.75,1,0.75,0.5,0.25,0]$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?%5B0%2C0.25%2C0.5%2C0.75%2C1%2C0.75%2C0.5%2C0.25%2C0%5D">

This is based on <!-- $ Number\ of\ bigrams = Length\ of\ sentence - 1$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?Number%5C%20of%5C%20bigrams%20%3D%20Length%5C%20of%5C%20sentence%20-%201">

### Results
The metrics were measured in two ways, each with both the [`sumeval`](https://pypi.org/project/sumeval/) and [`rouge`](https://pypi.org/project/rouge/) packages for each document and then averaged over all documents.
1) The average ROUGE scores with all the available gold standard summaries were taken (on-average performance) for each document.
2) The maximum ROUGE score of the available gold summaries (of a document) was taken to be the ROUGE score (for that document) (max performance). 

##### Maximum Performance, Rouge Package
| **Algorithm**       | **ROUGE-1** | **ROUGE-2** | **ROUGE-L** |
| ------------------- | ----------- | ----------- | ----------- |
| Centroid (TFIDF)    | 0.149       | 0.025       | 0.087       |
| Luhn                | 0.172       | 0.040       | 0.106       |
| LSA                 | 0.192       | 0.034       | 0.131       |
| LexRank             | 0.265       | 0.080       | 0.196       |
| TextRank            | 0.308       | 0.099       | 0.251       |
| BigramRank          | 0.328       | 0.134       | 0.292       |
| BigramRank (Ends)   | **0.338**   | **0.140**   | **0.303**   |
| BigramRank (Middle) | 0.301       | 0.100       | 0.255       |

![MaxRouge](assets/MaxRougeresults.png)

##### Average Performance, Rouge Package
| **Algorithm**       | **ROUGE-1** | **ROUGE-2** | **ROUGE-L** |
| ------------------- | ----------- | ----------- | ----------- |
| Centroid (TFIDF)    | 0.104       | 0.011       | 0.058       |
| Luhn                | 0.119       | 0.018       | 0.068       |
| LSA                 | 0.134       | 0.016       | 0.086       |
| LexRank             | 0.186       | 0.037       | 0.127       |
| TextRank            | 0.213       | 0.042       | 0.166       |
| BigramRank          | **0.227**   | 0.060       | 0.191       |
| BigramRank (Ends)   | 0.224       | **0.064**   | 0.193       |
| BigramRank (Middle) | 0.219       | 0.042       | **0.205**   |

![AverageRouge](assets/AverageRougeresults.png)

##### Maximum Performance, SumEval Package
| **Algorithm**       | **ROUGE-1** | **ROUGE-2** | **ROUGE-L** |
| ------------------- | ----------- | ----------- | ----------- |
| Centroid (TFIDF)    | 0.135       | 0.026       | 0.112       |
| Luhn                | 0.156       | 0.043       | 0.133       |
| LSA                 | 0.202       | 0.049       | 0.179       |
| LexRank             | 0.290       | 0.104       | 0.257       |
| TextRank            | **0.373**   | 0.122       | 0.335       |
| BigramRank          | 0.356       | 0.122       | 0.335       |
| BigramRank (Ends)   | 0.354       | **0.128**   | **0.337**   |
| BigramRank (Middle) | 0.317       | 0.096       | 0.297       |

![MaxSumEval](assets/Maxsumevalresults.png)

##### Average Performance, SumEval Package
| **Algorithm**       | **ROUGE-1** | **ROUGE-2** | **ROUGE-L** |
| ------------------- | ----------- | ----------- | ----------- |
| Centroid (TFIDF)    | 0.089       | 0.011       | 0.073       |
| Luhn                | 0.105       | 0.019       | 0.092       |
| LSA                 | 0.138       | 0.020       | 0.122       |
| LexRank             | 0.189       | 0.038       | 0.166       |
| TextRank            | **0.248**   | **0.053**   | **0.229**   |
| BigramRank          | 0.241       | 0.051       | 0.228       |
| BigramRank (Ends)   | 0.235       | 0.052       | 0.222       |
| BigramRank (Middle) | 0.219       | 0.042       | 0.205       |

![AverageSumEval](assets/Averagesumevalresults.png)

### Deep Learning
Coming soon!
