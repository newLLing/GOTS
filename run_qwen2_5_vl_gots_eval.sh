#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
  --model qwen2_5_vl_mmtok \
  --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct,device=cuda,selector_type=gots,retain_ratio=0.2,min_pixels=1605632,max_pixels=1605632 \
  --tasks gqa,mme,pope,mmbench_en_dev,scienceqa,ocrbench,textvqa_val,chartqa,docvqa_val,realworldqa,mathvista_testmini \
  --batch_size 1
