"""Small CPU checks of the packaged main path; no real LM/GPU experiment."""
import ast,importlib,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sager2.common import load_config,model_identity
class MainPackageTests(unittest.TestCase):
    def test_config_and_actual_bundled_datasets(self):
        from sager2.check import check
        for p in sorted((ROOT/'configs').glob('*.json')):
            with self.subTest(model=p.stem):
                cfg=load_config(p)
                if cfg['training_data'] is None:
                    self.assertIn(cfg['model_key'],('llama13b','llama30b'))
                    with self.assertRaisesRegex(FileNotFoundError,'--training-data'):check(cfg,False)
                else:self.assertTrue(check(cfg,False)['passed'])
    def test_no_old_project_imports(self):
        for p in (ROOT/'src/sager2').rglob('*.py'):
            tree=ast.parse(p.read_text(encoding='utf-8'))
            for n in ast.walk(tree):
                if isinstance(n,ast.ImportFrom) and n.level==0:
                    self.assertFalse((n.module or '').split('.')[0] in {'sager','accelerated_v1','common','variant_solver','opt_ops','angle_body'},str(p))
    def test_all_main_modules_import(self):
        for n in ['common','check','probe','models','data','extract','train','metrics','decode','editor_base','editor_newline','editor','runner','worker','supervisor','cli','backends.probe','backends.angle_body','backends.window_ops','backends.window_solver']:
            with self.subTest(module=n):importlib.import_module('sager2.'+n)
    def test_probe_math_on_actual_six_model_checkpoints(self):
        import numpy as np,torch
        if not list((ROOT/'checkpoints').rglob('*.pt')):self.skipTest('Trained checkpoints were removed after verified1199-prompt alignment')
        from sager2.probe import Probe
        from sager2.backends.probe import GPUProbe
        rng=np.random.default_rng(20261027)
        for p in sorted((ROOT/'configs').glob('*.json')):
            cfg=load_config(p);state=torch.load(Path(cfg['probe_dir'])/'block15.pt',map_location='cpu',weights_only=True)
            ref=Probe(state);gpu=GPUProbe.from_state(state,device='cpu')
            z=rng.normal(size=(4,cfg['expected_dimension']));z/=np.linalg.norm(z,axis=1,keepdims=True)
            actual=gpu.score(torch.tensor(z,dtype=torch.float64)).numpy();expected=np.array([ref.score(x) for x in z])
            np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-11)
    def test_all_six_refresh20_and_no_per_step_backend(self):
        for p in sorted((ROOT/'configs').glob('*.json')):
            self.assertEqual(load_config(p)['editing']['candidate_refresh_interval'],20)
        for name in ['editor_legacy.py','backends/legacy_solver.py','backends/angle_kernel.py']:
            self.assertFalse((ROOT/'src/sager2'/name).exists())
    def test_top100_candidate_selection_supports_all_three_hidden_sizes(self):
        import numpy as np,torch
        from unittest.mock import patch
        from sager2.backends import window_solver as module
        from sager2.backends.probe import GPUProbe
        rng=np.random.default_rng(20261027)
        for d in (4096,5120,6656):
            with self.subTest(hidden_size=d):
                z=rng.normal(size=d);z/=np.linalg.norm(z)
                w=rng.normal(size=(32,d));b=rng.normal(size=32);v=rng.normal(size=32)
                probe=GPUProbe(w,b,v,0.,device='cpu');search=module.GPUSearch(probe)
                self.assertEqual(search.d,d)
                g=w.T@(v*((w@z+b)>0))
                anchors=np.unique(np.r_[np.argsort(-np.abs(g))[:2],np.argsort(-np.abs(z))[:2]])
                pool=sorted({(min(int(a),j),max(int(a),j)) for a in anchors for j in range(d) if a!=j})
                gains=np.array([abs(z[i]*g[j]-z[j]*g[i]) for i,j in pool])
                expected=[pool[i] for i in np.argsort(-gains)[:100]]
                captured={}
                def capture(*args):
                    captured['i']=args[11].numpy().ravel();captured['j']=args[12].numpy().ravel();return ()
                with patch.object(module,'frozen_window',side_effect=capture):
                    module._ten(torch.tensor(z[None,:]),torch.zeros((1,d),dtype=torch.bool),torch.zeros(1,dtype=torch.long),torch.ones(1,dtype=torch.bool),
                        torch.tensor(800),torch.zeros(1,dtype=torch.long),torch.zeros(1,dtype=torch.long),torch.zeros(1,dtype=torch.long),
                        probe.W,probe.b,probe.v,probe.c,torch.arange(d),search.phi,True,False,probe.W.T.contiguous(),torch.zeros(()),True,False)
                self.assertEqual(list(zip(captured['i'],captured['j'])),expected)
    def test_native_cache_decoder_against_HF_small_random_model(self):
        import torch
        from types import SimpleNamespace
        from transformers import LlamaConfig,LlamaForCausalLM,GenerationConfig
        from sager2.models import Adapter
        from sager2.decode import generate_batch
        torch.manual_seed(7)
        cfg=LlamaConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=4,max_position_embeddings=64,bos_token_id=1,eos_token_id=2,pad_token_id=0)
        model=LlamaForCausalLM(cfg).eval();model.set_attn_implementation('eager')
        class Tokenizer:
            eos_token_id=2;pad_token_id=0
            def __call__(self,prompt,return_tensors=None):return SimpleNamespace(input_ids=torch.tensor([[1,3,4]]))
            def decode(self,ids,skip_special_tokens=True):return ' '.join(str(x) for x in ids if x not in [0,1,2])
        with torch.no_grad():
            manual=generate_batch(model,Tokenizer(),['test'],adapter=Adapter(model),max_new_tokens=4)['rows'][0]['new_token_ids']
            hf=model.generate(input_ids=torch.tensor([[1,3,4]]),attention_mask=torch.ones((1,3),dtype=torch.long),generation_config=GenerationConfig(do_sample=False,max_new_tokens=4,use_cache=True,bos_token_id=1,eos_token_id=2,pad_token_id=0,repetition_penalty=1.))[0,3:].tolist()
        self.assertEqual(manual,hf)
if __name__=='__main__':unittest.main()
