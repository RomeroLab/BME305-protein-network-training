"""Check masking and padding semantics without running notebook training."""
import ast
import json
from pathlib import Path
import unittest
import torch
from torch import nn
from torch.nn import functional as F
import lightning.pytorch as pl

ROOT = Path(__file__).parent
notebook = json.loads((ROOT / '02_Masked_language_model.ipynb').read_text())
namespace = dict(torch=torch, nn=nn, F=F, pl=pl, MASK=20, PAD=21, IGNORE=-100)
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if source.startswith(('def mask_tokens', 'mask_probability =', 'class MaskedProteinModel')):
            definitions = [node for node in ast.parse(source).body if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
            exec(compile(ast.Module(body=definitions, type_ignores=[]), '<notebook definitions>', 'exec'), namespace)
mask_tokens = namespace['mask_tokens']
MaskedProteinModel = namespace['MaskedProteinModel']


class NotebookContracts(unittest.TestCase):
    def test_padding_is_never_selected_and_labels_retain_originals(self):
        tokens = torch.tensor([[0, 1, 2, 21, 21], [3, 4, 21, 21, 21]])
        corrupted, labels = mask_tokens(tokens, probability=1, generator=torch.Generator().manual_seed(7))
        self.assertTrue(torch.equal(corrupted[tokens == 21], tokens[tokens == 21]))
        self.assertTrue((labels[tokens == 21] == -100).all())
        self.assertTrue(torch.equal(labels[tokens != 21], tokens[tokens != 21]))
        a = mask_tokens(tokens, generator=torch.Generator().manual_seed(19))
        b = mask_tokens(tokens, generator=torch.Generator().manual_seed(19))
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(a, b)))

    def test_at_least_one_target_per_sequence(self):
        tokens = torch.tensor([[0, 1, 21], [2, 3, 21]])
        _, labels = mask_tokens(tokens, probability=0)
        self.assertTrue(((labels != -100).sum(dim=1) == 1).all())

    def test_extra_padding_does_not_change_real_residue_predictions(self):
        torch.manual_seed(7)
        model = MaskedProteinModel(length=8, d_model=16, n_heads=4, n_layers=2).eval()
        with torch.inference_mode():
            short = model(torch.tensor([[0, 20, 2]]))
            padded = model(torch.tensor([[0, 20, 2, 21, 21, 21]]))
        torch.testing.assert_close(short, padded[:, :3], atol=1e-5, rtol=1e-5)

    def test_saved_sequences_are_unique_standard_and_in_length_range(self):
        import csv
        with (ROOT / 'data/uniref50_small.csv').open() as f:
            rows = list(csv.DictReader(f))
        sequences = [r['sequence'] for r in rows]
        self.assertEqual(len(rows), 4000)
        self.assertEqual(len(set(sequences)), 4000)
        self.assertTrue(all(40 <= len(s) <= 120 and set(s) <= set('ACDEFGHIKLMNPQRSTVWY') for s in sequences))
        split = (set(sequences[:3200]), set(sequences[3200:3600]), set(sequences[3600:]))
        self.assertTrue(all(not a.intersection(b) for i, a in enumerate(split) for b in split[i+1:]))


if __name__ == '__main__':
    unittest.main()
