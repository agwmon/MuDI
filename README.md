# MuDI
This is an official implementation of paper 'Identity Decoupling for Multi-Subject Personalization of Text-to-Image Models'.
* Arxiv: [link](https://arxiv.org/abs/2404.04243)
* Project page: [link](https://mudi-t2i.github.io/)

## Installation
```
pip install diffusers[torch] transformers peft wandb scipy

# for automatic masking
pip install git+https://github.com/facebookresearch/segment-anything.git
```

## Data preparation
Our training is based on DreamBooth and an additional augmentation method. For training, a segmentation mask is necessary, and the same goes for a prior preservation dataset.

### Personalization dataset
*TODO* - automatic mask generation

Automatic mask generation for the personalization dataset is available, but we recommend manually creating masks with precision using the Segment-Anything model.

In `dataset/category/actionfigure_2_and_dog0/metadata.jsonl`, we provide an example of our experiment setting.
```
{"id": {"a": "olis harry potter toy", "b": "hta corgi"}}
{"id": "a", "file_name": "a00.jpg", "mask_path": "mask_a00.jpg", "text": "A photo of a olis harry potter toy set on a polished wooden desk, with the sleek edge of a laptop and mouse in view."}
{"id": "a", "file_name": "a01.jpg", "mask_path": "mask_a01.jpg", "text": "A photo of a olis harry potter toy displayed on a smooth, dark surface with a vibrant blue cup in the background."}
{"id": "a", "file_name": "a02.jpg", "mask_path": "mask_a02.jpg", "text": "A photo of a olis harry potter toy standing confidently on a black surface with a white wall in the backdrop."}
{"id": "a", "file_name": "a03.jpg", "mask_path": "mask_a03.jpg", "text": "A photo of a olis harry potter toy captured against the contrast of a deep black tabletop and a striking blue cup in the background."}
{"id": "a", "file_name": "a04.jpg", "mask_path": "mask_a04.jpg", "text": "A photo of a olis harry potter toy strategically positioned on a desk with an intricate wood grain pattern and office supplies in soft focus behind."}
{"id": "b", "file_name": "b00.jpg", "mask_path": "mask_b00.jpg", "text": "A photo of a hta corgi with a backdrop of soft-hued cherry blossoms."}
{"id": "b", "file_name": "b01.jpg", "mask_path": "mask_b01.jpg", "text": "A photo of a hta corgi with a serene sky and flowering branches above."}
{"id": "b", "file_name": "b02.jpg", "mask_path": "mask_b02.jpg", "text": "A photo of a hta corgi against a vibrant orange backdrop and delicate flowers."}
{"id": "b", "file_name": "b03.jpg", "mask_path": "mask_b03.jpg", "text": "A photo of a hta corgi before a backdrop of cherry blossoms and a terracotta wall."}
{"id": "b", "file_name": "b04.jpg", "mask_path": "mask_b04.jpg", "text": "A photo of a hta corgi in an urban park setting with blurred foliage."}
```
The 'id' key in the first line only needs to be different from each other, and the value is used in the prompt for the seg-mix sample.
### Prior dataset
We found that using [DCO loss](https://github.com/kyungmnlee/dco) showed better performance than Dreambooth, **without the need for constructing a prior-preservation dataset**. Therefore, we used DCO loss instead in our code. 

However, all the experiments reported in our paper were conducted with Dreambooth using prior-preservation loss. If you want to re-implement, please check our automatic prior dataset generation pipeline. 
<details>
<summary>Prepare prior dataset</summary>
<div markdown="1">

We provide an automatic mask generation pipeline for the prior dataset. The prior mask does not need to be very accurate.
```
python generate_prior.py --gen_class $CLASS --gen_mask
```
In `dataset/reg/actionfigure_2_and_dog0/class_metadata.jsonl`, we provide an example of our experiment setting.

*TODO* - automatic data preperation pipeline

</div>
</details>


## Seg-Mix
In our experiments, about 18GB of GPU VRAM was used, and it is possible to run on < 15GB VRAM using --gradient_checkpoint.
```
VAL_PROMPT="olis harry potter toy and hta corgi playing together, floating on the pool."
DATASET_NAME="actionfigure_2_and_dog0"
SEG_MIX_PROB=0.3
SEG_MIX_START_STEP=0

accelerate launch train_segmix_lora_sdxl.py --instance_data_dir="dataset/category/"$DATASET_NAME \
    --pretrained_model_name_or_path=$SDXL_PATH \
    --pretrained_vae_model_name_or_path=$VAE_PATH \
    --output_dir=output/$DATASET_NAME"_p"$SEG_MIX_PROB"_s"$SEG_MIX_START_STEP \
    --validation_prompt="$VAL_PROMPT" --report_to=wandb --resolution=1024 --train_batch_size=1 --gradient_accumulation_steps=2 \
    --checkpointing_steps=20000 --max_train_steps=2010 --validation_epochs=3 --save_steps=200 --lr_warmup_steps=0 --seed=42 --rank=32 --learning_rate=1e-4 \
    --segmix_prob=$SEG_MIX_PROB --segmix_start_step=$SEG_MIX_START_STEP --relative_scale=0.0 --soft_alpha 1.0
```

## Inference
We provide an example of our Seg-Mix trained model. (olis harry potter toy & hta corgi)
[Google Drive](https://drive.google.com/file/d/1qNaZjf7pA-odpBwALbAaceZ06rmS0xAi/view?usp=sharing)


Please see inference_demo.ipynb

## TODO
- [X] Detect_and_Compare metric
- [ ] Automatic mask generation
- [X] with other training method (dco)
- [X] more than three concepts

## Bibtex
https://arxiv.org/abs/2404.04243
```
@misc{jang2024identity,
      title={Identity Decoupling for Multi-Subject Personalization of Text-to-Image Models}, 
      author={Sangwon Jang and Jaehyeong Jo and Kimin Lee and Sung Ju Hwang},
      year={2024},
      eprint={2404.04243},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
      
