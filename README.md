# Brain Tumor Segmentation Model

A PyTorch implementation of a modified U-Net architecture for semantic segmentation of brain tumor. This model seamlessly integrates **ResNeXt**, **Vision Transformers**, **ASPP**, and **Attention Gates** to capture both fine-grained local details and long-range global dependencies.


## Architecture Overview

The model follows an **Encoder-Decoder (U-Net style)** structure with a powerful bottleneck for context enhancement:

### 1. Encoder (Backbone)

- **ResNeXt-50:** Serves as the primary feature extractor.
- **Intermediate CNN Blocks:** After each encoder stage, additional convolution layers refine feature maps before they are passed to the next stage or skip connections.
- **Dropout2D:** Applied at each stage to prevent overfitting on spatial feature maps.

### 2. Bottleneck (Global Context)

- **Transformer Blocks:** A sequence of 6 Transformer layers (Multi-Head Attention) processes the flattened feature maps from the final encoder stage. This allows the model to model relationships between distant pixels.

- **ASPP (Atrous Spatial Pyramid Pooling):** Captures multi-scale information using multiple parallel dilated convolutions with different rates.

### 3. Decoder

- **Attention Gates:** Instead of simple concatenation, skip connections from the encoder are filtered through Attention Gates. These gates use the decoder's current state to "highlight" relevant features in the encoder maps and suppress noise.
- **Transpose Convolutions:** Used for learned upsampling to recover spatial resolution.

## Project Contents

- **BrainTumorSegmentation.ipynb** — Complete implementation, training, and evaluation code