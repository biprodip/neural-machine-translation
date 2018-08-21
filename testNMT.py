# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 00:12:29 2018

@author: biprodip
"""

# Inference models for testing

# Encoder inference model
encoder_model_inf = Model(encoder_input, encoder_states)

# Decoder inference model
decoder_state_input_h = Input(shape=(30,))
decoder_state_input_c = Input(shape=(30,))
decoder_input_states = [decoder_state_input_h, decoder_state_input_c]

decoder_out, decoder_h, decoder_c = decoder_LSTM(decoder_input,initial_state=decoder_input_states)
decoder_states = [decoder_h , decoder_c]
decoder_out = decoder_dense(decoder_out)
decoder_model_inf = Model(inputs=[decoder_input] + decoder_input_states,
                          outputs=[decoder_out] + decoder_states )






def decode_seq(inp_seq):
    
    # Initial states value is coming from the encoder 
    states_val = encoder_model_inf.predict(inp_seq)
    
    targetSequence = np.zeros((1, 1, len(banChars)))
    targetSequence[0, 0, banDictCharToInd['\t']] = 1
    
    translated_sent = ''
    stop_condition = False
    
    while not stop_condition:
        
        decoder_out, decoder_h, decoder_c = decoder_model_inf.predict(x=[targetSequence] + states_val)
        
        max_val_index = np.argmax(decoder_out[0,-1,:])
        sampleBangCharacters = banDictIndToChar[max_val_index]
        translated_sent += sampleBangCharacters
        
        if ( (sampleBangCharacters == '\n') or (len(translated_sent) > maxBanLen)) :
            stop_condition = True
        
        targetSequence = np.zeros((1, 1, len(banChars)))
        targetSequence[0, 0, max_val_index] = 1
        
        states_val = [decoder_h, decoder_c]
        
    return translated_sent





for seq_index in range(10):
    inp_seq = engSenToken[seq_index:seq_index+1]
    translated_sent = decode_seq(inp_seq)
    print('-')
    print('Input sentence:', engSentences[seq_index])
    print('Decoded sentence:', translated_sent)

