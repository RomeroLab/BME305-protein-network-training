# BME305: training protein neural networks

Train neural networks to learn from protein sequences. These two dry labs cover supervised prediction of protein function and self-supervised prediction of masked amino acids. Work through the model definitions and training steps, then inspect learning curves and predictions on held-out sequences.

## Notebooks

| Notebook | Input | Training objective | Open |
| --- | --- | --- | --- |
| [Sequence to function](01_Sequence_to_function.ipynb) | GB1 variants and experimental functional scores | Predict a continuous score with an embedding and convolutional network | [Colab](https://colab.research.google.com/github/RomeroLab/BME305-protein-network-training/blob/main/01_Sequence_to_function.ipynb) |
| [Masked protein language model](02_Masked_language_model.ipynb) | Natural UniRef50 representative sequences | Recover selected residues using a bidirectional transformer encoder | [Colab](https://colab.research.google.com/github/RomeroLab/BME305-protein-network-training/blob/main/02_Masked_language_model.ipynb) |

## Getting started

Open a notebook in Colab, select **Runtime → Change runtime type → T4 GPU**, and run the cells in order. Use **File → Save a copy in Drive** to keep your edits and outputs.

Each notebook shows the model architecture, forward pass, loss, and optimizer. PyTorch Lightning handles backpropagation and the training loop. Training data update the weights, validation data select a checkpoint, and test data evaluate the selected model.

## Sequence-to-function regression

Learn a mapping from GB1 amino acid sequences to measured functional scores. The network combines a learned amino acid embedding, two convolutional layers, and a regression head.

Use 12,000 variants with an 80/10/10 training, validation, and test split. Inspect training and validation loss, compare test predictions with measured scores, and assess improvement over a baseline that predicts the training-set mean. Explore how training-set size, model capacity, and learning rate affect performance.

## Masked-language training

Learn to predict hidden amino acids from their sequence context using 4,000 natural UniRef50 representatives, 40–120 residues long. The model combines amino acid and position embeddings with PyTorch's built-in transformer encoder.

Select 15% of real residues for prediction. Replace 80% of those with `[MASK]`, replace 10% with a random amino acid, and leave 10% unchanged. Padding is excluded from attention and loss. Training masks change each batch; validation and test masks stay fixed.

Compare test predictions with an amino acid frequency baseline, then inspect the predicted distribution at one masked residue. Explore how masking, model size, and learning rate affect learning.

## Interpreting results

The GB1 split evaluates predictions within one protein's sequence landscape. The UniRef50 subset provides a sample of short natural proteins. When interpreting performance, consider the diversity of training sequences and how closely the test sequences are related to them.

Use validation data to compare settings, and reserve test data for the final evaluation. Save the selected model and token definitions for later experiments.

## References

- [GB1 dataset](https://github.com/RomeroLab/Seq2FxnNN).
- [UniRef](https://www.uniprot.org/help/uniref) and [sequence dataset details](data/uniref50_manifest.json). UniProt data are distributed under [CC BY 4.0](https://www.uniprot.org/help/license).
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html), [PyTorch TransformerEncoder](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html), and [BERT](https://arxiv.org/abs/1810.04805).
