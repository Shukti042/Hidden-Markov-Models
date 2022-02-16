# Hidden Markov Models

Did modeling with a Hidden Markov Model (HMM), and implemented the Viterbi and the Baum-Welch algorithm. HMMs are applied in many fields including speech and image processing, bioinfor- matics and finance.

## Climate pattern modeling

El Nin ̃o–Southern Oscillation (ENSO) is an irregular periodic variation in winds and sea surface temperatures over the tropical eastern Pacific Ocean, affecting the climate of much of the tropics and subtropics. The warming phase of the sea temperature is known as El Nin ̃o and the cooling phase as La Nin ̃a both of which have long-term persistence. El Nin ̃o years in a particular basin tend to be wetter than La Nin ̃a years.

We can observe whether or not it is an El Nin ̃o year based on rainfall in the tropical Pacific for the present. But we are interested in understanding past climate variation using tree ring widths. We can infer from the tree ring data (with some error) what the total precipitation might have been in each year of the tree’s life.

So, we have two hidden states representing El Nin ̃o and La Nin ̃a. The observed quantities are rainfall estimates (from tree ring width) for the past T years. Let’s assume for simplicity that our observations Yt can be modeled by Gaussian distributions i.e. f(Yt|Xt = 1) ∼ N(μ1,σ12) and f(Yt|Xt = 2) ∼ N(μ2,σ2).

## Dataset

Two files titled “data.txt” and “parameters.txt” should be given containing the rainfall data for the past T years and parameters for the HMM respectively.

## Input

- The “data.txt” file contains T rows each containing a number indicat- ing the rainfall for that year.
- The “parameters.txt” file contains the number of states n (2 in this case) in the first line. The next n lines provides the transition matrix P . The next line gives the means of the n Gaussian distributions and the last line lists the standard deviations of the n Gaussian distribu- tions.

This implementation is easy to extend to arbitrary number of states of and emission probability distributions.

## Output

- A file containing estimated states using the parameters provided in “parameters.txt”. This will contain T rows each containing the esti- mated state for that year.
- A file containing parameters learned using the Baum-Welch algorithm. The format will be the same as “parameters.txt”. Add the stationary distribution in the last line.
- A file containing estimated states using the learned parameters. This will contain T rows each containing the estimated state for that year.

## How to Run

```bash
$ python 1605042.py <data_file_path> <parameter_file_path>
```

