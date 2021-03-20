config = {

    'SEED' : 1234,

    'datafile' : 'english_python_data.txt',
    'src_params' : {
        'init_token' : '<sos>',
        'eos_token' : '<eos>',
        'lower' : True,
        'batch_first' : True
    },
    'trg_params' : {
        'init_token' : '<sos>',
        'eos_token' : '<eos>',
        'lower' : False,
        'batch_first' : True
    },
    'src_min_freq' : 1,
    'trg_min_freq' : 1,
    'src_vocab_file' : 'src_vocab.pickle',
    'trg_vocab_file' : 'trg_vocab.pickle',

    # train / test iterators params
    'train_split' : 0.85,
    'sort' : False,
    'batch_size' : 32,
    'device' : 'cuda',

    # ENCODER DECODER params
    'HID_DIM' : 256,
    'ENC_LAYERS' : 3,
    'DEC_LAYERS' : 3,
    'ENC_HEADS' : 8,
    'DEC_HEADS' : 8,
    'ENC_PF_DIM' : 512,
    'DEC_PF_DIM' : 512,
    'ENC_DROPOUT' : 0.1,
    'DEC_DROPOUT' : 0.1,

    
    'LEARNING_RATE' : 0.0005,
    'N_EPOCHS' : 20,
    'CLIP' : 1,

    'min_train_loss_model' : 'min_train_loss_model.pt',
    'min_val_loss_model' : 'min_val_loss_model.pt',
    'weights' : 'min_train_loss_model.pt',
}