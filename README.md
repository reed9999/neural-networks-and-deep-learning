# Exploring Neural Networks by Hacking Neilsen's Implementation of MNIST Character Recognition

[Start here](https://github.com/reed9999/gdelt-demo/blob/master/Start-here.ipynb).

If you have Jupyter Notebook installed, run `jupyter notebook` and then via the Web GUI access `Start-here.ipynb`. 

This repository is a fork of [code samples](https://github.com/mnielsen/neural-networks-and-deep-learning)
accompanying Michael Nielsen's book [*Neural Networks
and Deep Learning*](http://neuralnetworksanddeeplearning.com).
I include it here to demonstrate my curiosity, and in particular how I play 
around with code to help me better understand one particular aspect of ML that caught my attention.

## Trying it out, status and next steps
This is messy code, but it should be self-contained. 

First do the standard `pip install -r requirements.txt` as needed, perhaps in a virtual environment, to install dependencies.

Then running ```python main.py``` from the src directory
should work to get some predictions after jimmying some of the starting weights,
which is the point of this exercise. 

Next steps: 
- Parameter so we can run in "classic mode" or in whatever I want to call the 
jimmied version.
- Save results! No need to run every time.
- Report results! Much better than copy/paste output.
- Bugs:
  - `/home/philip/code/neural-networks-and-deep-learning/src/network.py:182: RuntimeWarning: overflow encountered in exp
  return 1.0/(1.0+np.exp(-z))`

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
