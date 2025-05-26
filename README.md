<h1 align="center">SPA-RL-Agent</h1>
<p align="center">
  <a href=""><img src="https://img.shields.io/badge/arXiv-arXiv%20Preprint-B31B1B?style=flat&logo=arxiv&logoColor=white" alt="arXiv Paper"></a>
  &nbsp;
  <a href="https://github.com/WangHanLinHenry/SPA-RL-Agent"><img src="https://img.shields.io/badge/Homepage-Project%20Page-brightgreen?style=flat&logo=github" alt="Homepage"></a>
</p>

The repository contains the codes for our paper "[SPA-RL: Reinforcing LLM Agents via Stepwise Progress Attribution]()"

![image](./assets/spa_rl_framework.png)

This paper introduces **Stepwise Progress Attribution (SPA)**, a novel reward redistribution framework that provides fine-grained intermediate rewards by decomposing delayed rewards into incremental, per-step contributions, enabling more effective reinforcement learning for complex, multi-step agent tasks.

### ✨Key advantages include:
1. **Fine-grained Reward Redistribution**: Effectively decomposes delayed rewards into intermediate rewards, reflecting incremental progress toward task completion.  
2. **Effective Multi-turn RL Training**: Utilizes these intermediate rewards in PPO, achieving outstanding performance on complex long-horizon tasks.



## 🎉News
- [2025.05.26] 🚀SPA-RL-Agent Repo launched!

## 📝Contents

- [Setup](#setup)
  - [Installation](#installation)
  - [Environment Setup](#environment-setup)
- [Usage](#method)
  - [Base Agent SFT Training](#base-agent-sft-training)
  - [Environment Exploration](#environment-exploration)
  - [Progress Estimator Training](#progress-estimator-training)
  - [Step Rewards Annotation](#step-rewards-annotation)
  - [RL Training](#rl-training)
  - [Evaluation](#evaluation)
- [Baselines](#baselines)
  - [GRPO](#grpo)
  - [RAGEN](#ragen)

## Acknowledgement


## 🌹 Citation
If you find our work useful in your research please consider citing our paper:
```


```

