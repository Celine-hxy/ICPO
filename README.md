# ICPO: In-Context Steered Policy Optimization


The source code of the paper **Think Outside the Policy: In-Context Steered Policy Optimization**.

We base our code on `verl` and implement ICPO as a recipe extension (`verl/recipe/icpo`).

### Installation
We followed the official installation guide of [verl](https://github.com/volcengine/verl) to install the package from scratch.
```bash
conda create -n verl python==3.10
conda activate verl

cd verl
pip install --no-deps -e .

pip install datasets==4.0.0
pip install math_verify
```

We have provided a `requirements.txt` file showing our local environment (CUDA 12.4). We do not recommend installing from this file directly since it serves as a reference for our local environment only.

### 1. Data Preparation

Please refer to the Python scripts in `./prepare_data` to generate the required data files. Commands for preparing the training and demo datasets are as follows:

```bash
# Prepare train data
python ./prepare_data/openr1_math_220k-8192.py

# Prepare demonstration data
python ./prepare_data/math_dataset.py
python ./prepare_data/convert_pkl_to_demo_cache_pkl-math.py
```

### 2. GRPO / ICPO Training
Setup your path of `HOME`, `STORAGE` and your `WANDB_API_KEY` in `verl/recipe/icpo/scripts`. Then

```bash
bash ./verl/recipe/icpo/scripts/baseline/Qwen3-1.7b/GRPO_qwen3-1.7b.sh

bash ./verl/recipe/icpo/scripts/baseline/Qwen3-1.7b/ICPO_qwen3-1.7b_math-demo-icpo.sh
bash ./verl/recipe/icpo/scripts/baseline/Qwen3-1.7b/ICPO_qwen3-1.7b_math-demo-icpo-rs.sh
```
The checkpoints will be saved in `$HOME/checkpoints/ICPO/$exp_name`.


### 3. Evaluation

After training, please convert your checkpoint before evaluation:

```bash
bash ./verl/recipe/icpo/scripts/merge_ckpt.sh $your_checkpoint_path
```

For the specific parameters, we follow [LUFFY](https://github.com/ElliottYan/LUFFY). Please refer to the appendix of our paper for details.

## Acknowledgement

Our code is based on the [verl](https://github.com/volcengine/verl) framework for training, and the ICPO implementation is inspired by the [LUFFY](https://github.com/ElliottYan/LUFFY) project. We use the open-source [LIMO](https://github.com/GAIR-NLP/LIMO) codebase for evaluation.

## Citation

If you find our model, data, or evaluation code useful, please consider citing our paper.
```text
@misc{huang2025thinkoutsidepolicyincontext,
      title={Think Outside the Policy: In-Context Steered Policy Optimization}, 
      author={Hsiu-Yuan Huang and Chenming Tang and Weijie Liu and Saiyong Yang and Yunfang Wu},
      year={2025},
      eprint={2510.26519},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.26519}, 
}
```



