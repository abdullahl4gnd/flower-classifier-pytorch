# Flower Classifier

A CNN trained from scratch in PyTorch to classify the 102 flower types in the Oxford 102 Flowers dataset. It doesn't use any pretrained weights.

Test accuracy: 93.49% (best validation accuracy: 94.46%).

## Dataset

The dataset has 8,189 images in 102 classes. The notebook downloads it automatically the first time you run it (about 330 MB). I split it into 5,733 training, 1,228 validation and 1,228 test images with a fixed random seed.

## Model

The model has five blocks. Each block is two conv layers with batch norm and ReLU, followed by max pooling. After the blocks there is global average pooling, dropout (0.3) and a linear layer with 102 outputs.

## Training

- Optimizer: AdamW, learning rate 1e-3, weight decay 0.01
- Scheduler: OneCycleLR, max learning rate 3e-3
- Loss: cross-entropy with label smoothing 0.1
- Batch size 16, 50 epochs, mixed precision on GPU
- Augmentation: random resized crop, horizontal flip, rotation, color jitter
- The checkpoint with the best validation accuracy is saved as best_model.pt

I trained it on an RTX 3050 laptop GPU. On CPU it works but is much slower.

## How to run

1. Clone the repo and create a virtual environment.
2. If you have an NVIDIA GPU, install PyTorch with CUDA first (pick the right command at pytorch.org). For example:
   `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`
3. Install the rest: `pip install -r requirements.txt`
4. Open `App3.ipynb` and run all cells.

## Notes

The dataset class skips images that are corrupted or too small instead of crashing. To test this, the notebook makes a copy of the data in `corrupted_flower_data` and damages a few images there. The original data is not changed.

## Files

- `App3.ipynb`: data loading, model, training and evaluation
- `helper_utils.py`: prints the dataset folder tree
- `best_model.pt`: trained weights
- `requirements.txt`

## Dataset credit

Nilsback and Zisserman, Automated Flower Classification over a Large Number of Classes, ICVGIP 2008.