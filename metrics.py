import nltk
from nltk.translate.bleu_score import SmoothingFunction
from pyformer import translate_sentence

def calculate_bleu(data, src_field, trg_field, model, device, max_len = 500):
    
    trgs = []
    pred_trgs = []
    
    for datum in data:
        
        src = vars(datum)['src']
        trg = vars(datum)['trg']
        
        pred_trg, _ = translate_sentence(src, src_field, trg_field, model, device, max_len)
        
        #cut off <eos> token
        pred_trg = pred_trg[:-1]
        
        pred_trgs.append(pred_trg)
        trgs.append(trg)
        # print(len(trg), len(pred_trg))
        if len(trg) < 2 or len(pred_trg) < 2:
            print(src)
    # print(trgs[0], pred_trgs[0])
    assert(len(trgs) == len(pred_trgs))
    smooth_fn = SmoothingFunction().method4
    return nltk.translate.bleu_score.corpus_bleu(trgs, pred_trgs, smoothing_function = smooth_fn)