# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 00:12:29 2018

@author: biprodip
"""
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np


#load data
allLines = open('english_bangla.txt', encoding='utf-8').read().split('\n')
engChars = set()
banChars = set()
engSentences = []
banSentences  = []
totSamples = 35 #10000

for singleLine in range(totSamples):
    #Each line is splited in two halves based on the first \t
    #1st half[0] is appended into engLines
    #2nd half[1] is stored into banLines 
    #Bangla sentences: Start of sentence: '\t'  end of the sentence:'\n'
    
    engHalf = str(allLines[singleLine]).split('\t')[0]
    banHalf = '\t' + str(allLines[singleLine]).split('\t')[1] + '\n'
    engSentences.append(engHalf)
    banSentences.append(banHalf)
    
    for eChar in engHalf:
        if (eChar not in engChars):
            engChars.add(eChar)
            
    for bChar in banHalf:
        if (bChar not in banChars):
            banChars.add(bChar)
            
banChars = sorted(list(banChars))
engChars = sorted(list(engChars))




# dictionary to index each english character - key is index and value is english character
engDictIndToChar = {}

# dictionary to get english character given its index - key is english character and value is index
engDictCharToInd = {}

for k, v in enumerate(engChars):
    engDictIndToChar[k] = v
    engDictCharToInd[v] = k
    
# dictionary to index each bangla character - key is index and value is bangla character
banDictIndToChar = {}

# dictionary to get bangla character given its index - key is bangla character and value is index
banDictCharToInd = {}
for k, v in enumerate(banChars):
    banDictIndToChar[k] = v
    banDictCharToInd[v] = k
      
maxEngLen= max([len(line) for line in engSentences])
maxBanLen = max([len(line) for line in banSentences])


engSenToken = np.zeros(shape = (totSamples,maxEngLen,len(engChars)), dtype='float32')
banSenToken = np.zeros(shape = (totSamples,maxBanLen,len(banChars)), dtype='float32')
target = np.zeros((totSamples, maxBanLen, len(banChars)),dtype='float32')


# Vectorize the english and bangla sentences

for i in range(totSamples):
    for k,ch in enumerate(engSentences[i]):
        engSenToken[i,k,engDictCharToInd[ch]] = 1
        
    for k,ch in enumerate(banSentences[i]):
        banSenToken[i,k,banDictCharToInd[ch]] = 1

        # decoder_target will be ahead by one timestep and will not include the start character.
        if k > 0:
            target[i,k-1,banDictCharToInd[ch]] = 1
            
            
# Encoder model

encoder_input = Input(shape=(None,len(engChars)))
encoder_LSTM = LSTM(30,return_state = True)
encoder_outputs, encoder_h, encoder_c = encoder_LSTM (encoder_input)
encoder_states = [encoder_h, encoder_c]


# Decoder model

decoder_input = Input(shape=(None,len(banChars)))
decoder_LSTM = LSTM(30,return_sequences=True, return_state = True)
decoder_out, _ , _ = decoder_LSTM(decoder_input, initial_state=encoder_states)
decoder_dense = Dense(len(banChars),activation='softmax')
decoder_out = decoder_dense (decoder_out)



model = Model(inputs=[encoder_input, decoder_input],outputs=[decoder_out])

# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit(x=[engSenToken,banSenToken], 
          y=target,
          batch_size=5,
          epochs=500,
          validation_split=0.2)
