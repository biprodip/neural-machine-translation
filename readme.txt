LSTM maintains short-term-memory in cell state over long time steps.
Overcomes overcoming the vanishing gradient problem.


Gates can be thought of as a "conventional" artificial neuron, as in a multi-layer (or feedforward) neural network: that is, they compute an 
activation (using an activation function) of a weighted sum. Intuitively, they can be thought as regulators of the flow of values that goes 
through the connections of the LSTM; hence the denotation "gate"

Figure 26 of https://arxiv.org/pdf/1702.07800.pdf

http://blog.echen.me/2017/05/30/exploring-lstms/
Gates:
Forgetting mechanism: If a scene ends, for example, the model should forget the current scene location, the time of day, and reset any scene-specific information; however, if a character dies in the scene, it should continue remembering that he's no longer alive. 
Thus, we want the model to learn a separate forgetting/remembering mechanism: when new inputs come in, it needs to know which beliefs to keep or throw away.

Saving mechanism. When the model sees a new image, it needs to learn whether any information about the image is worth using and saving. 
Maybe your mom sent you an article about the Kardashians, but who cares?

So when new a input comes in, the model first forgets any long-term information it decides it no longer needs. Then it learns which parts of the new input are worth using, and saves them into its long-term memory.
Focusing long-term memory into working memory. Finally, the model needs to learn which parts of its long-term memory are immediately useful. For example, Bob's age may be a useful piece of information to keep in the long term (children are more likely to be crawling, adults are more likely to be working), but is probably irrelevant if he's not in the current scene. So instead of using the full long-term memory all the time, it learns which parts to focus on instead.


Mathematical intuition:
https://medium.com/@kangeugine/long-short-term-memory-lstm-concept-cb3283934359
Sigmoid specifically, is used as the gating function for the 3 gates(in, out, forget) in LSTM, since it outputs a value between 0 and 1, it can either let no
flow or complete flow of information throughout the gates. On the other hand, to overcome the vanishing gradient problem, we need a function whose second
derivative can sustain for a long range before going to zero. 
Tanh is a good function with the above property. [https://stackoverflow.com/questions/40761185/what-is-the-intuition-of-using-tanh-in-lstm]



The encoder is used to encode the data into representations
and decoder is used to make sequential predictions. Attention mechanism is used to locate
a region of the representation for predicting the label in current time step.

Epoch 500/500
16/16 [==============================] - 6s 385ms/step - loss: 0.0419 - val_loss: 3.7001

Input sentence: ﻿Go.
Decoded Bangla sentence: যাও

-
Input sentence: Run!
Decoded sentence: পালাও!

-
Input sentence: Run!
Decoded sentence: পালাও!

-
Input sentence: Wow!
Decoded sentence: ওয়াও!

-
Input sentence: Fire!
Decoded sentence: আগুন!

-
Input sentence: Help!
Decoded sentence: বাঁচাও!