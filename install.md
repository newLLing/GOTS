# GOTS Installation (dependency order)

## 1. Create environment

```bash
conda create -n mmtok python=3.12 -y
conda activate mmtok
```

## 2. Install PyTorch and base deps

```bash
# PyTorch (tested with 2.8.0 + CUDA 12.8 on H100 and A6000; newer versions may cause issues)
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

## 3. Install GOTS (from this repo root)

```bash
pip install -e .
```

Verify installation (should print `mmtok installed successful`):

```bash
python -c "import mmtok; print('mmtok installed successful')"
```


## 4. Install lmms-eval (for evaluation)


```bash

cd lmms-eval
# Recommend
uv pip install -e ".[all]"
```

To use GOTS with `lmms-eval`, you simply need to **wrap the model instance** in its corresponding evaluation file within `lmms-eval/lmms_eval/models/`. 

#### Step 1: Add the Import

Add the GOTS integration at the top of your model file (e.g., `qwen2_5_vl.py`):

```python
from mmtok import mmtok
# For Qwen2.5-VL specifically, use the specialized wrapper
from mmtok.qwen.qwen2_5_vl_mmtok import mmtok_qwen2_5_vl
```

#### Step 2: Wrap the Model

Locate the section where the model and tokenizer are initialized (typically in the `__init__` or `_create_model` method) and apply the wrapper.

| Model | Implementation | Key Parameter |
| --- | --- | --- |
| **Qwen2.5-VL** | `self._model, self.processor = mmtok_qwen2_5_vl(...)` | `retain_ratio=0.1` (fraction, e.g., 10%) |

#### Step 3: Code Snippet

**For Qwen2.5-VL (`qwen2_5_vl.py`):**

```python
# GOTS: wrap both model and processor

self._model, self.processor = mmtok_qwen2_5_vl(
    self._model, 
    language_tokenizer=self._tokenizer, 
    processor=self.processor, 
    retain_ratio=0.1
)
```

## 5. Optional: Flash Attention

We tested that with the versions above (e.g. torch 2.8.0), Flash Attention 2 can be installed relatively easily. With newer torch versions, installing flash-attn may run into environment or build issues that you may need to resolve.

```bash
pip install flash-attn --no-build-isolation
```


```python
attn_implementation="flash_attention_2"
```

## 6. Optional: Qwen-VL

```bash
pip install qwen-vl-utils
```

If you see `get_model_name_from_path` is not defined, try `pip install transformers>=4.52.4`.

