# CS1671 Assignment 2

Jacob Emmerson

Due: 10/04/23

## About

This script was developed in WSL with an Ubuntu in a Conda ENV Python Version: 3.11.4

It features two classes:

`NgramModel(context_length, k)`

`NgramModelWithInterpolation(context_length, k)`

Where `k` is the value added for smoothing.

## Running the Code

To run the code, enter the following into the terminal with the python script.

    python3 ngram_skeleton.py

You will be prompted for an input file path for training, and the context length you wish to use. Running the script directly only demonstrates the model's ability to generate random text. If you wish to use the methods in the script, put the following into the head of a python script:

    from ngram_skeleton import *

## Unresolved Issues / Misc. Notes

As noted at the end of the report, the smoothing function allows for probabilities greater than 1 when k > N + |V| - 1

## Additional Resources

- https://en.wikipedia.org/wiki/Perplexity
- https://en.wikipedia.org/wiki/Additive_smoothing