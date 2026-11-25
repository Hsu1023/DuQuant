import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
import unittest,numpy as np,torch
from torch.nn.attention import SDPBackend,sdpa_kernel
from eval_lambada_ppl import score_words
from duquant_cuda.decoder import LlamaDecoderBlock
from duquant_cuda.model import LlamaInferenceModel
from test_decoder import small_weights

@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class LambadaTests(unittest.TestCase):
    @torch.inference_mode()
    def test_greedy_multitoken_and_loglikelihood(self):
        torch.manual_seed(55)
        model=LlamaInferenceModel([LlamaDecoderBlock(small_weights(),{},'fp16',max_position=64,kv_bits=16)],
            torch.randn(128,256,device='cuda',dtype=torch.float16)*.1,
            torch.ones(256,device='cuda',dtype=torch.float16),
            torch.randn(128,256,device='cuda',dtype=torch.float16)*.03,1e-5)
        requests=[]
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            for ctx,length in [([1,5,8],1),([2,4],3)]:
                sequence=ctx[:];target=[]
                for _ in range(length):
                    ids=torch.tensor([sequence],device='cuda');logits=model(ids,model.caches(1,len(sequence)))
                    t=int(logits.argmax(-1));target.append(t);sequence.append(t)
                requests.append(dict(context=ctx,continuation=target))
            bad=dict(context=requests[1]['context'],continuation=requests[1]['continuation'][:])
            bad['continuation'][-1]=(bad['continuation'][-1]+1)%128;requests.append(bad)
            ll,flags,_=score_words(model,requests,32,model.caches(3,32))
            expected=[]
            for r in requests:
                prefix=r['context'][:];total=0
                for t in r['continuation']:
                    ids=torch.tensor([prefix],device='cuda');logits=model(ids,model.caches(1,len(prefix)))
                    total+=logits.float().log_softmax(-1)[0,t].item();prefix.append(t)
                expected.append(total)
        np.testing.assert_array_equal(flags,[True,True,False])
        np.testing.assert_allclose(ll,expected,atol=.004,rtol=0)

if __name__=='__main__':unittest.main()
