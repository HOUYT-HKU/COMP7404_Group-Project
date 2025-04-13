## Project Structure

### Vit Structure
1. **`vit_patch_and_positional.py`**  
   This file contains the code for creating ViT's patch embeddings and positional embeddings. It defines how to split the input image into patches and how to apply positional encoding to these patches.

2. **`vit_transformer_block.py`**  
   This file implements the transformer block, which is the main component of ViT. It contains the multi-head self-attention mechanism and feed-forward layers used to process the input features.

3. **`vit_patch_vit.py`**  
   This file combines the patch embeddings and transformer blocks to form the complete ViT model. It includes the vision transformer architecture from patch embedding to final classification output.

4. **`vit_train.py`**  
   This is a demo file to showcase how to use the core ViT structure for training on small datasets like CIFAR-10. It contains training scripts and necessary configurations to run the ViT model.

### Train
1. **`Small-CIFAR10.py`**  
   This script trains the ViT model on CIFAR-10, a small dataset with 60,000 images. It uses the `vit_train.py` structure to train the model from scratch, providing a baseline comparison for other models such as ResNet-50.

2. **`Resnet-50.py`**  
   This script trains a ResNet-50 model on CIFAR-10. It is used as a baseline model for comparison with ViT. You can compare the accuracy of ViT and ResNet-50 to evaluate the performance on small datasets.

3. **`21k-pre.py`**  
   This script is for pretraining the ViT model using the ImageNet-21K dataset. By using a large dataset for pretraining, the model can learn better feature representations, which can be transferred to smaller datasets like CIFAR-10 and CIFAR-100.

4. **`fine-tune.py`**  
   This script fine-tunes the pretrained ViT model (from ImageNet-21K) on smaller datasets like CIFAR-100 or any other dataset you want to use. It allows you to take advantage of pretraining to improve performance on your specific task.

