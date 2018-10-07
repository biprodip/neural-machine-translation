# neural-machine-translation
LSTM based encoder decoder model of basic neural machine translation for English to Bangla translation
LSTM parameters are given in code.

The encoder is used to encode the data into representations and decoder is used to make sequential predictions. 
# testing
Epoch 500/500
16/16 [==============================] - 6s 385ms/step - loss: 0.0419 - val_loss: 3.7001

Input sentence: ﻿Go.
Decoded Bangla sentence: যাও

Input sentence: Run!
Decoded sentence: পালাও!

Input sentence: Wow!
Decoded sentence: ওয়াও!

Input sentence: Fire!
Decoded sentence: আগুন!

Input sentence: Help!
Decoded sentence: বাঁচাও!


# Dependencies
* Python
* Keras
* Numpy
