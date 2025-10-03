<img src="./figs/teaser.png" alt="Teaser Image" width="700">

# [arXiv] Is Exchangeability better than I.I.D to handle Data Distribution Shifts while Pooling Data for Data-scarce Medical image segmentation?

<p align="center">
  <strong>Ayush Roy</strong>¹ &middot; 
  <strong>Samin Enam</strong>¹ &middot;  
  <strong>Jun Xia</strong>¹ &middot; 
  <strong>Won Hwa Kim</strong>² &middot; 
  <strong>Vishnu Suresh Lokhande</strong>¹
</p>

<p align="center">
  ¹ University at Buffalo, SUNY &bull; ² POSTECH
</p>

## Abstract

Data scarcity is a major challenge in medical imaging, particularly for deep learning models. While data pooling (combining datasets from multiple sources) and data addition (adding more data from a new dataset) have been shown to enhance model performance, they are not without complications. Specifically, increasing the size of the training dataset through pooling or addition can induce distributional shifts, negatively affecting downstream model performance, a phenomenon known as the “Data Addition Dilemma”. While the traditional i.i.d. assumption may not hold in multi-source contexts, assuming exchangeability across datasets provides a more practical framework for data pooling. In this work, we investigate medical image segmentation under these conditions, drawing insights from causal frameworks to propose a method for controlling foreground-background feature discrepancies across all layers of deep networks. This approach improves feature representations, which are crucial in data-addition scenarios. Our method achieves state-of-the-art segmentation performance on histopathology and ultrasound images across five datasets, including a novel ultrasound dataset that we have curated and contributed. Qualitative results demonstrate more refined and accurate segmentation maps compared to prominent baselines across three model architectures. The code will be available on Github.

## Installation

Our code repo builds on top of the `ldm` environment of [`compvis/latent-diffusion`](https://github.com/CompVis/latent-diffusion).

Users can close the `CompVis/latent-diffusion` repo and install all requirements using the following commands.

```bash
git clone https://github.com/CompVis/latent-diffusion.git

cd latent-diffusion

conda env create -f environment.yaml
conda activate ldm
```

We additionally use the `diffusers` library for Textual Inversion training. Specifically we use the following packages:

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .

cd examples/textual_inversion
pip install -r requirements.txt

accelerate config
accelerate config default
```

## Watermark Detectors

AquaLoRA: Please refer to [AquaLoRA](./AquaLORA/README.md) for clear instructions to pre-train watermark detector.

HiDDeN: We utilize implementation by `Stable Signature` to pre-train HiDDeN watermark detector. More details are given in [hidden](./hidden/README.md) folder.

## Training Token Embeddings for watermarking

![Training Figure](./figs/method_final.jpeg)

We provide code for training token embeddings for watermarking in [TI_Black_Box](./TI_Black_Box/w_TI_48_bit_aquaLORA.py) folder.

All arguments used for training are present in `TI_Black_Box/train_TI_aquaLORA.sh` and training can be run using the following command.

```bash
bash train_TI_aquaLORA.sh
```

## Object Level Watermarking

![](./figs/main_qual_new.png)

We follow `google/prompt-to-prompt` to utilize attention maps for object level watermarking.

We provide `[./object-level-watermarking/p2p_ti_object_watermark.ipynb]` jupyter notebook for object level watermarking implementation.

Specifically, we use `AttentionReplace`:

```python
class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
      
    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)
```

![](./figs/object_output.png)

We also provide Attention Maps of the watermarking token in the figure below.

![](./figs/token_attn_map_supp-1.png)


# Acknowledgements

Our codebase is based on the following repos:

1. https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/README.md
2. https://github.com/facebookresearch/stable_signature
3. https://github.com/CompVis/latent-diffusion
4. https://github.com/Georgefwt/AquaLoRA
5. https://github.com/google/prompt-to-prompt/

# Citation


```bibtex
@misc{devulapally2025textencoderobjectlevelwatermarking,
      title={Your Text Encoder Can Be An Object-Level Watermarking Controller}, 
      author={Naresh Kumar Devulapally and Mingzhen Huang and Vishal Asnani and Shruti Agarwal and Siwei Lyu and Vishnu Suresh Lokhande},
      year={2025},
      eprint={2503.11945},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.11945}, 
}
```
