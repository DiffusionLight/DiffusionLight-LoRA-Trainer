# DiffusionLight-LoRA-Trainer: a code for training LoRA use in DIffusionLight
### [Project Page](https://diffusionlight.github.io/) | [Main Repository](https://github.com/DIffusionLight/DiffusionLight)

We provide a code and training set for training LoRA using in DiffusionLight.

## Table of contents
-----
  * [TL;DR](#Getting-started)
  * [Dataset](#Dataset)
  * [Installation](#Installation)
  * [Citation](#Citation)
------

### Getting started

```shell
bash final_train_cont_largeT@900.sh
```

wait over a night on Single RTX 3090 until it create folder `ThisIsTheFinal-lora-hdr-continuous-largeT@900/0_-5/checkpoint-2500` which we provide in main repository


### Dataset 

We use the [Text2Light](https://frozenburning.github.io/projects/text2light/) to generate HDR Environment map using the prompt which also generated by ChatGPT-3.5. 

However, you can download a pre-generated dataset at [OneDrive (133 GB)](https://vistec-my.sharepoint.com/:f:/g/personal/pakkapon_p_s19_vistec_ac_th/Es_xIFk67UBPkVdW25cwDScBdvNlsZYz8lZblO7ZGzqGNg?e=q1dUqd) which contain 1412 images pair of Low-FOV LDR and HDR Environment map

## Installation

Please follow the instructions in our main repository's [installation guide](https://github.com/DiffusionLight/DiffusionLight#Installation)


## Citation

```
@inproceedings{Phongthawee2023DiffusionLight,
    author = {Phongthawee, Pakkapon and Chinchuthakun, Worameth and Sinsunthithet, Nontaphat and Raj, Amit and Jampani, Varun and Khungurn, Pramook and Suwajanakorn, Supasorn},
    title = {DiffusionLight: Light Probes for Free by Painting a Chrome Ball},
    booktitle = {ArXiv},
    year = {2023},
}
```

## Visit us 🦉
[![Vision & Learning Laboratory](https://i.imgur.com/hQhkKhG.png)](https://vistec.ist/vision) [![VISTEC - Vidyasirimedhi Institute of Science and Technology](https://i.imgur.com/4wh8HQd.png)](https://vistec.ist/)
