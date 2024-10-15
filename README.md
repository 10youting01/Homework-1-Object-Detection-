# Homework-1-Object-Detection-
## Environment Details
* The environment requires the following libraries:
  * Python: 3.7 or higher
  * CUDA: Ensure GPU is available (nvidia-smi command to verify)
  * PyTorch: 1.10 or higher with CUDA support
  * Other Libraries:
  * transformers
  * pytorch-lightning
  * roboflow
  * supervision
  * timm
  * opencv-python
  * torchvision
    ```
    import torch
    torch.cuda.is_available()  # Should return True if GPU is available
    torch.__version__  # Check PyTorch version
    ```
## Installation
Model: https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-huggingface-detr-on-custom-dataset.ipynb
All of the required environment are added in the .ipynb file

## How to run the code
* Dataset Preparation
  * Ensure your dataset is in the COCO format (JSON annotations) and images are stored in train, valid, and test directories.
  * The structure of the dataset folders should look like this:
  * ```
    /train
    - /images
    - /labels
    - train_annotations_coco.json
    /valid
      - /images
      - /labels
      - valid_annotations_coco.json
      - valid_target.json
    /test
      - /images
      - test_annotations_coco.json
    eval_1009.py
    
    ```
  * Run the .ipynb file in colab or vscode
  * Evaluate the valid result in the terminal
  * ```
    python eval_1009.py valid/valid_predictions_resnet101.json valid_target.json
    ```
