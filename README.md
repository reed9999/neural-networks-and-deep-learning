# Exploring Neural Networks: Neilsen's Book and MNIST 

This repository is a fork of [code samples](https://github.com/mnielsen/neural-networks-and-deep-learning)
accompanying Michael Nielsen's online book [*Neural Networks
and Deep Learning*](http://neuralnetworksanddeeplearning.com).
The intent is to demonstrate my curiosity and impetus to hack around 
with other people's code to understand how something like NN really works.
I have read through chapter 2 and am working on chapter 3 of the book, so the questions involved 
in my "experiments" reflect where I am, in the middle of the book.
Because I learn best through an iterative, hands-on approach, there's value to documenting my
curiosity and trying to address it on my own, whether or not these questions are addressed later on..

## Running from scratch 

First do the standard `pip install -r requirements.txt` as needed (recommended: in a virtual 
environment) to install dependencies.

Then run ```python main.py``` from the src directory to execute the main functionality.

## The two experiments
Two questions have emerged in my mind and led me to do further experimentation:

1. If we quantify the probabilities of a "pulse" making it through all layers of neurons, we can
get a set of 28 x 28 maps of probabilities for each pixel in the images. **Does this map should produce something humans can identify as looking somewhat like 
the numerals 0, 1, 2, etc.?**

2. **If I tweak the initial weights to match intuition about, for example, "What an eight 
looks like," does that improve predictive power?** 

It seems like it should, because the initial 
weights have less learning to do, but it's not that simple. As in (1) I'm really interested in 
the cross product of all the matrices for all the different levels. As a first approximation, can 
I seed the weights in just one layer? Does this encourage the other layers to end up with something
like an identity matrix?

## Next steps 
(needs update)
- Parameter so we can run in "classic mode" or in whatever I want to call the 
jimmied version.
- Save results! No need to run every time.
- Report results! Much better than copy/paste output.
- Bugs:
  - `/home/philip/code/neural-networks-and-deep-learning/src/network.py:182: RuntimeWarning: overflow encountered in exp
  return 1.0/(1.0+np.exp(-z))` \[But note! I'm getting this running the original NN structure too, assuming I haven't accidentally broken something there.]

And that should be enough to make this "decent." I'd also like to: 

- Work on later chapters of Nielsen's ebook. (I got a bit sidetracked on the
    exploration reflected here.)


## Background
I was introduced to ML in some PhD classwork, but I was coming at it from
a social scientist's perspective. Thus I tend to think a lot about ML's claim 
(as I understand it) to be atheoretical and to be better 
at prediction without much *a priori* theory. Since it's fun
to be skeptical and to figure things out for oneself, I decided to play with
the NIST handwriting examples here.

This isn't intended as a finished project, just an example of exploration.
I wondered if, given that neural networks do a really good job with the NIST
task, wouldn't they do a slightly better job with just a little *a priori*
theoretical help? For example, I theorize that a digit with light pixels in
middle is disproportionately likely to be a zero because zeroes have a hole
in the middle. So instead of seeding initial values in the NN at random, 
wouldn't it be better to seed the layers in such a way that the pixels in 
the middle of the zero have coefficients suggesting lighter pixels?

But in fact, in my initial messing around, I found this wasn't really 
happening, at least not noticeably. Perhaps this bolsters the claims to 
atheoreticity. 




## Neilsen's original README notes

The code is written for Python 2.6 or 2.7. Michal Daniel Dobrzanski
has a repository for Python 3
[here](https://github.com/MichalDanielDobrzanski/DeepLearningPython35). I
will not be updating the current repository for Python 3
compatibility.

The program `src/network3.py` uses version 0.6 or 0.7 of the Theano
library.  It needs modification for compatibility with later versions
of the library.  I will not be making such modifications.

As the code is written to accompany the book, I don't intend to add
new features. However, bug reports are welcome, and you should feel
free to fork and modify the code.

## License

MIT License

Copyright (c) 2012-2015 Michael Nielsen

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
