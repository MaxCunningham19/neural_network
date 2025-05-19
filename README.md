# Neural Networks from Scratch

I made this project after watching [Andrej Karpathy's micrograd video](https://www.youtube.com/watch?v=VMj-3S1tku0). I was inspired to try to re-create this myself. I built a similar version that only uses simple operations i.e. I did not implement softmax or tan. This means sometimes the exponentials end up diffrenciating badly resulting in `nans`.

## Running

This was built on MacOS using Python 3.13

Install requirements using:

```
pip install -r requirements.txt
```

I created a simple script to generate some binary data and then used this to train the model. To run the model set up the data.csv with inputs labeledxn and outputs labeled yj. Then run:

```
python main.py
```

This will read from the data.csv and infer the shape of the inputs and outputs accordingly.
