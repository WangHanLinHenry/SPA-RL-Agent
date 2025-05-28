<h1 align="center">SPA-RL-Agent</h1>
<p align="center">
  <a href="https://arxiv.org/abs/2505.20732"><img src="https://img.shields.io/badge/arXiv-arXiv%20Preprint-B31B1B?style=flat&logo=arxiv&logoColor=white" alt="arXiv Paper"></a>
  &nbsp;
  <a href="https://github.com/WangHanLinHenry/SPA-RL-Agent"><img src="https://img.shields.io/badge/Homepage-Project%20Page-brightgreen?style=flat&logo=github" alt="Homepage"></a>
</p>

The repository contains the codes for our paper "[SPA-RL: Reinforcing LLM Agents via Stepwise Progress Attribution](https://arxiv.org/abs/2505.20732)"

<p align="center">
  <img src="./assets/spa_rl_framework.png" width="95%">
</p>

This paper introduces **Stepwise Progress Attribution (SPA)**, a novel reward redistribution framework that provides fine-grained intermediate rewards by decomposing delayed rewards into incremental, per-step contributions, enabling more effective reinforcement learning for complex, multi-step agent tasks (Webshop, ALFWorld and VirtualHome).

### ✨Key advantages include:
1. **Fine-grained Reward Redistribution**: Effectively decomposes delayed rewards into intermediate rewards, reflecting incremental progress toward task completion.  
2. **Effective Multi-turn RL Training**: Utilizes these intermediate rewards in PPO, achieving outstanding performance on complex long-horizon tasks.



## 🎉News
- [2025.05.28] 🔥 Release our paper on arXiv. See [here](https://arxiv.org/abs/2505.20732).
- [2025.05.26] 🚀 SPA-RL-Agent Repo launched!

## 📝Contents

- [Setup](#setup)
  - [Installation](#installation)
  - [Environment Setup](#environment-setup)
- [Usage](#method)
  - [Base Agent SFT Training](#base-agent-sft-training)
  - [Explored Trajectories Collection](#environment-exploration)
  - [Progress Estimator Training](#progress-estimator-training)
  - [Stepwise Progress Prediction](#step-rewards-annotation)
  - [RL Training](#rl-training)
  - [Evaluation](#evaluation)
- [Baselines](#baselines)
  - [GRPO](#grpo)
  - [RAGEN](#ragen)
  - ...

## ⚙️ Setup
```
# Create virtual environment for agentic evaluation
conda create -n SPA python=3.9
conda activate SPA
pip install -r requirements.txt

# Download webshop environment setting
cd envs/webshop
pip install -e .
python -m spacy download en_core_web_lg
conda install -y -c conda-forge openjdk=11

# Download data for WebShop environment
gdown https://drive.google.com/uc?id=1G_0ccLWn5kZE5rpeyAdh_YuoNzvBUjT9
gdown https://drive.google.com/uc?id=11zOUDkJSgGhYin9NxQtG8PVpDsika86y
unzip data.zip
mkdir search_index
unzip indexes.zip -d search_index/

# Download data for ALFWorld environment
cd ../..
cd eval_agent/data/alfworld
gdown https://drive.google.com/uc?id=1y7Vqeo0_xm9d3I07vZaP6qbPFtyuJ6kI
unzip alfworld_data.zip
```

Create another virtual environment for RL training due to package conflicts:
```
conda create -n RL_train python=3.10
conda activate RL_train
pip install -r ppo/requirements.txt
```

## ⛏️ Usage

### 🤖 Base Agent SFT Training

```
cd sft
# For ALFWorld environment
bash alfworld_llama3b.sh
# For Webshop environment
bash webshop_llama3b.sh
# For VirtualHome environment
bash virtualhome_llama3b.sh
```
⚠️ Note that the bash scripts provide the hyperparameters to reproduce our results. You should modify the settings, such as the model path, according to your own environment.

### 🕹️ Explored Trajectories Collection

```
cd ..
# For ALFWorld environment
bash exploration/alfworld/my_generate_response.sh
# For WebShop environment
bash exploration/webshop/my_generate_response_webshop.sh
```

### 📈 Progress Estimator Training

To orgainize the exploration data for progress estimator training, please run the following scripts firstly.
```
python prm/data_org.py
```
Then you could run the following script to train progress estimator.
```
deepspeed --include=localhost:0,1,2,3 prm/train_our_progress_model.py
```

### 🤷‍♂️ Stepwise Progress Prediction

```
python prm/inference_prm.py
```

### 💪🏽 RL Training

To organize the inference data for RL training, please run the following script first.
```
python prm/rl_data_org.py
```
Next, execute the following script to perform reinforcement learning training using LoRA:
```
conda activate rl
bash ppo/train_ppo.sh
```

### 🎮 Evaluation
Before evaluation, we need to merge the LoRA weights with the original LLM weights to obtain the final model:
```
python ppo/merge.py
```
Then, we would run the evaluation scripts:
```
# For ALFWorld environment
bash eval/llama3_2_3b_eval_alfworld.sh
# For WebShop environment
bash eval/llama3_2_3b_eval_webshop.sh
# For VirtualHome environment
bash eval/llama3_2_3b_eval_virtualhome.sh
```

### Running Baselines
TODO

## 🌹 Acknowledgement
Our code implementation is based on [ETO](https://github.com/Yifan-Song793/ETO) and [steptool](https://github.com/yuyq18/steptool). We thank them for their great work. 

Also very thankful for my wonderful co-authors: [Chak Tou Leong](https://cooperleong00.github.io/), [Jiashuo Wang](https://www4.comp.polyu.edu.hk/~csjwang/), [Jian Wang](https://iwangjian.github.io/), [Wenjie Li](https://www4.comp.polyu.edu.hk/~cswjli/).



## 🙏 Citation
If you find our work useful in your research please consider citing our paper:
```

```

