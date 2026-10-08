# BME305: training protein neural networks

Two Colab notebooks introduce supervised and self-supervised learning with protein sequences. Students work directly with model definitions, losses, optimizers, and PyTorch Lightning training steps, then inspect learning curves and held-out predictions.

## Notebooks

| Notebook | Input | Training objective | Open |
| --- | --- | --- | --- |
| [Sequence to function](01_Sequence_to_function.ipynb) | GB1 variants and experimental functional scores | Predict a continuous score with an embedding and convolutional network | [Colab](https://colab.research.google.com/github/RomeroLab/BME305-protein-network-training/blob/main/01_Sequence_to_function.ipynb) |
| [Masked protein language model](02_Masked_language_model.ipynb) | Natural UniRef50 representative sequences | Recover selected residues using a bidirectional transformer encoder | [Colab](https://colab.research.google.com/github/RomeroLab/BME305-protein-network-training/blob/main/02_Masked_language_model.ipynb) |

## Getting started

Open a notebook in Colab, select **Runtime → Change runtime type → T4 GPU**, and run the cells in order. Use **File → Save a copy in Drive** to keep your edits and outputs. Each notebook installs Lightning; Colab supplies PyTorch and the other standard packages. No pretrained model downloads are needed.

Both notebooks show the model architecture, forward pass, loss calculation, and optimizer. Lightning handles backpropagation and the training loop. Training data update weights, validation data select a checkpoint, and test data evaluate the selected model. Saved models include token definitions and hyperparameters.

## Sequence-to-function regression

This exercise adapts [RomeroLab/Seq2FxnNN](https://github.com/RomeroLab/Seq2FxnNN), using its GB1 variant dataset and a smaller, modern Lightning implementation. It retains the embedding → convolution → regression approach, with a learned amino acid embedding instead of the original physicochemical initialization.

The notebook samples 12,000 variants from the pinned source dataset and uses a reproducible 80/10/10 split. Score normalization uses only the selected training data. Students compare learning curves, test-set predictions, and a training-mean baseline, then vary training fraction or model settings.

GB1 variant labels omit the initial methionine of the full reference sequence. The conversion accounts for that indexing. Scores reflect the binding-selection experiment; the task is regression of the supplied score. Random held-out variants assess interpolation within this local sequence landscape, rather than transfer to new proteins.

## Masked-language training

The dataset contains 4,000 natural UniRef50 cluster representatives, 40–120 residues long. The saved snapshot excludes ambiguous amino acids, annotated fragments, and exact duplicate sequences. [The provenance record](data/uniref50_manifest.json) gives the source query, retrieval date, UniProt release, filtering, and checksum. It is a convenience sample of short sequences, not a uniform sample of UniRef50.

The model uses `nn.Embedding`, `nn.TransformerEncoderLayer`, and `nn.TransformerEncoder`. Real residues attend bidirectionally; padding is excluded from attention and loss. Fifteen percent of real positions are selected, with the BERT-style 80% mask / 10% random / 10% unchanged corruption scheme. Training masks change each batch; validation and test masks are fixed. Independently initialized encoder layers avoid starting with identical cloned weights.

Evaluation compares the transformer with amino acid frequencies estimated from the training set. It reports cross-entropy, selected-residue accuracy, and accuracy where the input was replaced by `[MASK]`. Students inspect one held-out masked residue and compare the full amino acid distributions.

This model illustrates protein language-model training from scratch on a small dataset. It does not reproduce the capabilities of a large pretrained PLM. Different UniRef50 representatives can still be evolutionarily related; a research evaluation would require a more deliberate homology-aware split and broader sequence coverage.

## Data and sources

- [Seq2FxnNN](https://github.com/RomeroLab/Seq2FxnNN), source revision `688d7bcf1455b9c7fd5fa7cda36089a340beca3d`, provides the GB1 dataset and original teaching example.
- [UniRef](https://www.uniprot.org/help/uniref) provides natural sequence representatives. UniProt data are distributed under [CC BY 4.0](https://www.uniprot.org/help/license); retain the attribution and provenance when reusing the subset.
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html), [PyTorch TransformerEncoder](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html), and [BERT](https://arxiv.org/abs/1810.04805) describe the training framework, encoder components, and masking objective.

## Development

The notebooks use Lightning 2.5.5 and Colab's installed PyTorch. To check masking, padding invariance, and the sequence subset without running training, install Lightning and run `python -m unittest -v test_notebooks`.
