"""
This is for training our model
    - SAVE_PATH                                 : This is a path to save our model's parameters
        models\{}check_point_{}.pth
        ㄴ{ } First : Keyword, Second : Epochs
    - LOAD_PATH                                 : This is a path to load our model's parameters
        models\{}check_point_{}.pth
        ㄴ{ } First : Keyword, Second : Epochs
    - EPOCHS                                    : The number of time to see the dataset
    - NUM_BATCH : Batch number                  : The number of data size to put in our model at once
    - SEQ_LEN                                   : The length of input x
    - INPUT_DIM                                 : The dimension of the input xt
    - HIDDEN_DIM                                : The dimension of the hidden state
    - NUM_LAYERS                                : The number of layers of LSTM
"""