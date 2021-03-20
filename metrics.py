from torchtext.data.metrics import bleu_score
from pyformer import translate_sentence

def calculate_bleu(data, src_field, trg_field, model, device, max_len = 520):
    
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
    assert(len(trgs) == len(pred_trgs))
    return bleu_score(pred_trgs, trgs)