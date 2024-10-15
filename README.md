# CT-CHAT

Welcome to the official repository for **CT-CHAT**, a cutting-edge visual-language chat model designed specifically for 3D chest CT volumes. CT-CHAT provides an open-source codebase and pre-trained models, utilizing [CT-CLIP](https://github.com/ibrahimethemhamamci/CT-CLIP) and a VQA (Visual Question Answering) dataset adapted from [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE), making it accessible to researchers worldwide. The VQA dataset and model weights are available via the [HuggingFace repository](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE).

<p align="center">
  <img src="figures/CTCHAT-demo.gif" width="100%">
</p>



## System Requirements

Before you get started, ensure that your environment meets the following requirements:

- **Python version**: > 3.12.4
- **Necessary dependencies**: Install CT-CLIP’s dependencies by following the instructions in the [CT-CLIP repository](https://github.com/ibrahimethemhamamci/CT-CLIP).
- **Additional libraries**: Ensure that the following libraries are installed:
  - PyTorch v2.4.0
  - CUDA v12.4
  - SciPy v1.14.0
  - Torchvision v0.19.0
  - Scikit-learn v1.2.2
  - Pandas v2.2.2
  - Transformers v4.44.0
  - NumPy v1.26.4

### Hardware Requirements

- **For training**: 
  - Small models: Minimum of 2 A100 GPUs with 80GB VRAM.
  - Large models (80B Llama 3.1): Minimum of 4 A100 GPUs.
  
- **For inference**: 
  - Large models: At least 2 A100 GPUs.
  - Smaller models: 1 A100 GPU.

## Training

To train the model, follow the provided scripts. It's crucial to run the training data through the image encoder to generate embeddings prior to training. Use the provided [Encoder Script](https://github.com/ibrahimethemhamamci/CT-CHAT/blob/main/llava/serve/encode_script.py) as a reference for encoding a single image. Note that this differs from the latent-saving process in CT-CLIP; the outputs must be saved before latent projection. Update the training scripts with the correct path to the saved encodings and other necessary configurations.

## Inference

For inference, refer to the [serve scripts](llava/serve). To perform CLI-based inference, the validation data must first be encoded similarly to the training data. After encoding, adjust the required paths in the CT-CHAT validation scripts for CLI inference.

For GUI-based inference, run the following commands:

```bash
python -m llava.serve.controller --host 0.0.0.0 --port 10000
python -m llava.serve.gradio_web_server --controller http://localhost:10000 --model-list-mode reload
python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path "path_to_model" --model-base "path_to_model"
```

## Pretrained Models

We offer pre-trained models for several LLMs, trained on the VQA dataset described in our paper. You can download them from the links below:

- **CT-CHAT (Llama 3.1 70B)**: [Download Here](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE)
- **CT-CHAT (Llama 3.1 8B)**: [Download Here](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE)
- **CT-CHAT (Vicuna)**: [Download Here](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE)
- **CT-CHAT (Mistral)**: [Download Here](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE)

## VQA Dataset

The VQA dataset has been derived from the CT-RATE data using the Llama 3.1 80B model with the scripts provided [here](./VQA_dataset). Short-answer questions have been sampled from the [RadGenome Chest CT dataset](https://huggingface.co/datasets/RadGenome/RadGenome-ChestCT). The dataset is available in the [CT-RATE HuggingFace repository](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE).

## Citing Us

If you use CT-CHAT, CT-CLIP, or our CT-RATE dataset in your research, please cite [our paper](https://arxiv.org/abs/2403.17834). 

## License
We are committed to fostering innovation and collaboration in the research community. To this end, all elements of CT-RATE, CT-CLIP, and CT-CHAT are released under a [Creative Commons Attribution (CC-BY-NC-SA) license](https://creativecommons.org/licenses/by-nc-sa/4.0/). This licensing framework ensures that our contributions can be freely used for non-commercial research purposes, while also encouraging contributions and modifications, provided that the original work is properly cited and any derivative works are shared under similar terms.



## Acknowledgements
We would like to express our sincere gratitude to the following works, whose contributions were invaluable to our research. Our VQA dataset includes a subset of data from [RadGenome Chest CT](https://arxiv.org/abs/2404.16754). Additionally, our CT-CHAT model is a 3D adaptation of the [LLaVA](https://arxiv.org/pdf/2304.08485) model for CT volumes. CT-CHAT leverages CT-ViT architecture as the vision encoder which is introduced as part of [GenerateCT](https://arxiv.org/abs/2305.16037). We are deeply appreciative of these researchers for their outstanding open-source contributions. If you use our VQA data or CT-CHAT model in your work, we kindly ask that you also cite the related works to acknowledge their impact.





