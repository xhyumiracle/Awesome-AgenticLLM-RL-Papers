# Awesome-AgenticLLM-RL-Papers
This is the Official repo for the survey paper: The Landscape of Agentic Reinforcement Learning for LLMs: A Survey

[ArXiv – https://arxiv.org/abs/2509.02547](https://arxiv.org/abs/2509.02547)  

[HuggingFace – https://huggingface.co/papers/2509.02547](https://huggingface.co/papers/2509.02547)

## Citation
```bibtex
@misc{zhang2025landscapeagenticreinforcementlearning,
      title={The Landscape of Agentic Reinforcement Learning for LLMs: A Survey}, 
      author={Guibin Zhang and Hejia Geng and Xiaohang Yu and Zhenfei Yin and Zaibin Zhang and Zelin Tan and Heng Zhou and Zhongzhi Li and Xiangyuan Xue and Yijiang Li and Yifan Zhou and Yang Chen and Chen Zhang and Yutao Fan and Zihu Wang and Songtao Huang and Yue Liao and Hongru Wang and Mengyue Yang and Heng Ji and Michael Littman and Jun Wang and Shuicheng Yan and Philip Torr and Lei Bai},
      year={2025},
      eprint={2509.02547},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2509.02547}, 
}
```


## Sec2.7 Agentic RL: Algorithms

Clip corresponds to preventing the policy ratio from moving too far from 1 for ensuring stable updates.  
KL penalty corresponds to penalizing the KL divergence between the learned policy and the reference policy for ensuring alignment.  

| Method | Year | Objective Type | Clip | KL Penalty | Key Mechanism | Signal | Link | Resource |
|--------|------|----------------|------|-------------|---------------|--------|------|----------|
| **_PPO family_** |||||||||
| PPO | 2017 | Policy gradient | Yes | No | Policy ratio clipping | Reward | [Paper](https://arxiv.org/abs/1707.06347) | - |
| VAPO | 2025 | Policy gradient | Yes | Adaptive | Adaptive KL penalty + variance control | Reward + variance signal | [Paper](https://arxiv.org/abs/2504.05118) | - |
| PF-PPO | 2024 | Policy gradient | Yes | Yes | Policy filtration | Noisy reward | [Paper](https://arxiv.org/abs/2409.06957) | [Code](https://github.com/DtYXs/verl/tree/pf-ppo) |
| VinePPO | 2024 | Policy gradient | Yes | Yes | Unbiased value estimates | Reward | [Paper](https://arxiv.org/abs/2410.01679) | [Code](https://github.com/McGill-NLP/VinePPO) |
| PSGPO | 2024 | Policy gradient | Yes | Yes | Process supervision | Process Reward | [Paper](https://openreview.net/forum?id=Cn5Z0MUPZT) | - |
| **_DPO family_** |||||||||
| DPO | 2024 | Preference optimization | No | Yes | Implicit reward related to the policy | Human preference | [Paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/a85b405ed65c6477a4fe8302b5e06ce7-Paper-Conference.pdf) | - |
| β-DPO | 2024 | Preference optimization | No | Adaptive | Dynamic KL coefficient | Human preference | [Paper](https://openreview.net/forum?id=ZfBuhzE556) | [Code](https://github.com/junkangwu/beta-DPO) |
| SimPO | 2024 | Preference optimization | No | Scaled | Use avg log-prob of a sequence as implicit reward | Human preference | [Paper](https://openreview.net/forum?id=3Tzcot1LKb) | [Code](https://github.com/princeton-nlp/SimPO) |
| IPO | 2024 | Implicit preference | No | No | LLMs as preference classifiers | Preference rank | [Paper](https://proceedings.mlr.press/v238/gheshlaghi-azar24a.html) | - |
| KTO | 2024 | Knowledge transfer optimization | No | Yes | Teacher stabilization | Teacher-student logit | [Paper](https://openreview.net/forum?id=iUwHnoENnl) | [Code](https://github.com/ContextualAI/HALOs) [Model](https://huggingface.co/collections/ContextualAI/archangel-65bd45029fa020161b052430) |
| ORPO | 2024 | Online regularized preference optimization | No | Yes | Online stabilization | Online feedback reward | [Paper](https://arxiv.org/abs/2403.07691) | [Code](https://github.com/xfactlab/orpo) [Model](https://huggingface.co/kaist-ai/mistral-orpo-alpha) |
| Step-DPO | 2024 | Preference optimization | No | Yes | Step-wise supervision | Step-wise preference | [Paper](https://arxiv.org/abs/2406.18629) | [Code](https://github.com/dvlab-research/Step-DPO) [Model](https://huggingface.co/collections/xinlai/step-dpo-6682e12dfbbb2917c8161df7) |
| LCPO | 2025 | Preference optimization | No | Yes | Length preference with limited data/training | Reward | [Paper](https://arxiv.org/abs/2508.10164) | - |
| **_GRPO family_** |||||||||
| GRPO | 2025 | Policy gradient under group-based reward | Yes | Yes | Group-based relative reward to eliminate value estimates | Group-based reward | [Paper](https://arxiv.org/abs/2501.12948) | - |
| DAPO | 2025 | Surrogate of GRPO's | Yes | Yes | Decoupled clip + dynamic sampling | Dynamic group-based reward | [Paper](https://arxiv.org/abs/2503.14476) | [Code](https://github.com/BytedTsinghua-SIA/DAPO) [Model](https://huggingface.co/BytedTsinghua-SIA/DAPO-Qwen-32B) [Website](https://dapo-sia.github.io/) |
| GSPO | 2025 | Surrogate of GRPO's | Yes | Yes | Sequence-level clipping, rewarding, optimization | Smooth group-based reward | [Paper](https://arxiv.org/pdf/2507.18071) | - |
| GMPO | 2025 | Surrogate of GRPO's | Yes | Yes | Geometric mean of token-level rewards | Margin-based reward | [Paper](https://arxiv.org/abs/2507.20673) | [Code](https://github.com/callsys/GMPO) |
| ProRL | 2025 | Same as GRPO's | Yes | Yes | Reference policy reset | Group-based reward | [Paper](https://arxiv.org/pdf/2505.24864) | [Model](https://huggingface.co/nvidia/Nemotron-Research-Reasoning-Qwen-1.5B) |
| Posterior-GRPO | 2025 | Same as GRPO's | Yes | Yes | Reward only successful processes | Process-based reward | [Paper](https://arxiv.org/pdf/2508.05170) | - |
| Dr.GRPO | 2025 | Unbiased GRPO objective | Yes | Yes | Eliminate bias in optimization | Group-based reward | [Paper](https://arxiv.org/pdf/2503.20783) | [Code](https://github.com/sail-sg/understand-r1-zero) [Model](https://huggingface.co/collections/sail/oat-zero-understanding-r1-zero-like-training-67dcdb07b9f3eb05f1501c4a) |
| Step-GRPO | 2025 | Same as GRPO's | Yes | Yes | Rule-based reasoning rewards | Step-wise reward | [Paper](https://arxiv.org/pdf/2503.12937) | [Code](https://github.com/jingyi0000/R1-VL) [Model](https://huggingface.co/collections/jingyiZ00/r1-vl-67d8e2cbcbe40158b0a45e74) |
| SRPO | 2025 | Same as GRPO's | Yes | Yes | Two-staged history-resampling | Reward | [Paper](https://arxiv.org/pdf/2504.14286) |  [Model](https://huggingface.co/Kwaipilot/SRPO-Qwen-32B) |
| GRESO | 2025 | Same as GRPO's | Yes | Yes | Pre-rollout filtering | Reward | [Paper](https://arxiv.org/abs/2506.02177) | [Code](https://github.com/Infini-AI-Lab/GRESO/) [Website](https://infini-ai-lab.github.io/GRESO/) |
| StarPO | 2025 | Same as GRPO's | Yes | Yes | Reasoning-guided actions for multi-turn interactions | Group-based reward | [Paper](https://arxiv.org/abs/2504.20073) | [Code](https://github.com/RAGEN-AI/RAGEN) [Website](https://ragen-ai.github.io/) |
| GHPO | 2025 | Policy gradient | Yes | Yes | Adaptive prompt refinement | Reward | [Paper](https://arxiv.org/pdf/2507.10628) | [Code](https://github.com/hkgc-1/GHPO) |
| Skywork R1V2 | 2025 | GRPO with hybrid reward signal | Yes | Yes | Selective sample buffer | Multimodal reward | [Paper](https://arxiv.org/pdf/2504.16656) | [Code](https://github.com/SkyworkAI/Skywork-R1V) [Model](https://huggingface.co/Skywork/Skywork-R1V2-38B) |
| ASPO | 2025 | GRPO with shaped advantage | Yes | Yes | Clipped bias to advantage | Group-based reward | [Paper](https://arxiv.org/pdf/2508.19201) | [Code]() [Model]() |
| TreePo | 2025 | Same as GRPO's | Yes | Yes | Self-guided rollout, reduced compute burden | Group-based reward | [Paper](https://arxiv.org/pdf/2508.17445) | [Code](https://github.com/multimodal-art-projection/TreePO) [Model](https://huggingface.co/collections/m-a-p/treepo-68ad9a7c078e83cb49cd9b2d) [Website](https://m-a-p.ai/TreePO/) |
| EDGE-GRPO | 2025 | Same as GRPO's | Yes | Yes | Entropy-driven advantage + error correction | Group-based reward | [Paper](https://arxiv.org/pdf/2507.21848) | [Code](https://github.com/ZhangXJ199/EDGE-GRPO) [Model](https://huggingface.co/collections/Zhang199/edge-grpo-688974025917352b5e335752) |
| DARS | 2025 | Same as GRPO's | Yes | No | Multi-stage rollout for hardest problems | Group-based reward | [Paper](https://arxiv.org/pdf/2508.13755) | [Code](https://github.com/yangzhch6/DARS) [Model](https://huggingface.co/collections/yangzhch6/dars-68a6c755262b9867f420c386) |
| CHORD | 2025 | Weighted GRPO + SFT | Yes | Yes | Auxiliary supervised loss | Group-based reward | [Paper](https://arxiv.org/pdf/2508.11408) | [Code](https://github.com/modelscope/Trinity-RFT/tree/main/examples/mix_chord) |
| PAPO | 2025 | Surrogate of GRPO's | Yes | Yes | Implicit Perception Loss | Group-based reward | [Paper](https://arxiv.org/pdf/2507.06448) | [Code](https://github.com/MikeWangWZHL/PAPO) [Model](https://huggingface.co/collections/PAPOGalaxy/papo-qwen-686d92dd3d43b1ce698f851a) [Website](https://mikewangwzhl.github.io/PAPO/) |
| Pass@k Training | 2025 | Same as GRPO's | Yes | Yes | Pass@k metric as reward | Group-based reward | [Paper](https://arxiv.org/abs/2508.10751) | [Code](https://github.com/RUCAIBox/Passk_Training) |


## Sec4.1 Task: Search & Research Agent

| Method | Category | Base LLM | Link | Resource |
|--------|----------|----------|------|----------|
| **_Open Source Methods_** |||||
| DeepRetrieval | External | Qwen2.5-3B-Instruct, Llama-3.2-3B-Instruct | [Paper](https://arxiv.org/pdf/2503.00223) | [Code](https://github.com/pat-jj/DeepRetrieval) |
| Search-R1 | External | Qwen2.5-3B/7B-Base/Instruct | [Paper](https://arxiv.org/abs/2503.09516) | [Code](https://github.com/PeterGriffinJin/Search-R1) |
| R1-Searcher | External | Qwen2.5-7B, Llama3.1-8B-Instruct | [Paper](https://arxiv.org/abs/2503.05592) | [Code](https://github.com/RUCAIBox/R1-Searcher) |
| R1-Searcher++ | External | Qwen2.5-7B-Instruct | [Paper](https://arxiv.org/abs/2505.17005) | [Code](https://github.com/RUCAIBox/R1-Searcher-plus) |
| ReSearch | External | Qwen2.5-7B/32B-Instruct | [Paper](https://arxiv.org/abs/2503.19470) | [Code](https://github.com/Agent-RL/ReCall/tree/re-search) |
| StepSearch | External | Qwen2.5-3B/7B-Base/Instruct | [Paper](https://arxiv.org/abs/2505.15107) | [Code](https://github.com/Zillwang/StepSearch) |
| WebDancer | External | Qwen2.5-7B/32B, QWQ-32B | [Paper](https://arxiv.org/abs/2505.22648) | [Code](https://github.com/Alibaba-NLP/WebAgent/tree/main/WebDancer) |
| WebThinker | External | QwQ-32B, DeepSeek-R1-Distilled-Qwen-7B/14B/32B, Qwen2.5-32B-Instruct | [Paper](https://arxiv.org/abs/2504.21776) | [Code](https://github.com/sunnynexus/WebThinker) |
| WebSailor | External | Qwen2.5-3B/7B/32B/72B | [Paper](https://arxiv.org/abs/2507.02592) | [Code](https://github.com/Alibaba-NLP/WebAgent/tree/main/WebSailor) |
| AutoTIR | External | Qwen2.5-7B-Instruct | [Paper](https://arxiv.org/pdf/2507.21836) | [Code](https://github.com/weiyifan1023/AutoTIR) |
| WebWatcher | External | Qwen2.5-VL-7B/32B | [Paper](https://arxiv.org/pdf/2508.05748) | [Code](https://github.com/Alibaba-NLP/WebAgent/tree/main/WebWatcher) |
| ASearcher | External | Qwen2.5-7B/14B, QwQ-32B | [Paper](https://arxiv.org/pdf/2508.07976) | [Code](https://github.com/inclusionAI/ASearcher) |
| ZeroSearch | Internal | Qwen2.5-3B/7B-Base/Instruct | [Paper](https://arxiv.org/abs/2505.04588) | [Code](https://github.com/Alibaba-NLP/ZeroSearch) |
| SSRL | Internal | Qwen2.5-1.5B/3B/7B/14B/32B/72B-Instruct, Llama-3.2-1B/8B-Instruct, Llama-3.1-8B/70B-Instruct, Qwen3-0.6B/1.7B/4B/8B/14B/32B | [Paper](https://arxiv.org/abs/2508.10874) | [Code](https://github.com/TsinghuaC3I/SSRL) |
| **_Closed Source Methods_** |||||
| OpenAI Deep Research | External | OpenAI Models | [Blog](https://openai.com/index/introducing-deep-research/) | [Website](https://chatgpt.com/) |
| Perplexity’s DeepResearch | External | - | [Blog](https://www.perplexity.ai/hub/blog/introducing-perplexity-deep-research) | [Website](https://www.perplexity.ai/) |
| Google Gemini’s DeepResearch | External | Gemini | [Blog](https://gemini.google/overview/deep-research/) | [Website](https://gemini.google.com/) |
| Kimi-Researcher | External | Kimi K2 | [Blog](https://moonshotai.github.io/Kimi-Researcher/) | [Website](https://www.kimi.com/) |
| Grok AI DeepSearch | External | Grok3 | [Blog](https://grokaimodel.com/deepsearch/) | [Website](https://grok.com/) |
| Doubao with Deep Think | External | Doubao | [Blog](https://seed.bytedance.com/en/special/doubao_1_5_pro) | [Website](https://www.doubao.com/chat/) |



---

## Sec4.2 Task: Code Agent
| Method | RL Reward Type | Base LLM | Link | Resource |
|--------|----------------|----------|------|----------|
| **_RL for Code Generation_** ||||| 
| AceCoder | Outcome | Qwen2.5-Coder-7B-Base/Instruct, Qwen2.5-7B-Instruct | [Paper](https://arxiv.org/abs/2502.01718) | [Code](https://github.com/TIGER-AI-Lab/AceCoder) |
| DeepCoder-14B | Outcome | Deepseek-R1-Distilled-Qwen-14B | [Blog](https://pretty-radio-b75.notion.site/DeepCoder-A-Fully-Open-Source-14B-Coder-at-O3-mini-Level-1cf81902c14680b3bee5eb349a512a51) | [Code](https://github.com/agentica-project/rllm) |
| RLTF | Outcome | CodeGen-NL 2.7B, CodeT5 | [Paper](https://openreview.net/forum?id=hjYmsV6nXZ) | [Code](https://github.com/Zyq-scut/RLTF) |
| CURE | Outcome | Qwen2.5-7B/14B-Instruct, Qwen3-4B | [Paper](https://arxiv.org/abs/2506.03136) | [Code](https://github.com/Gen-Verse/CURE) |
| Absolute Zero | Outcome | Qwen2.5-7B/14B, Qwen2.5-Coder-3B/7B/14B, Llama-3.1-8B | [Paper](https://arxiv.org/abs/2505.03335) | [Code](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner) |
| MSRL | Outcome | Qwen2.5-VL-7B-Instruct | [Paper](https://arxiv.org/abs/2508.13587) | [Code](https://github.com/DocTron-hub/MSRL) |
| StepCoder | Process | DeepSeek-Coder-Instruct-6.7B | [Paper](https://aclanthology.org/2024.acl-long.251/) | [Code](https://github.com/Ablustrund/APPS_Plus) |
| Process Supervision-Guided PO | Process | - | [Paper](https://openreview.net/forum?id=Cn5Z0MUPZT) | - |
| CodeBoost | Process | Qwen2.5-Coder-7B-Instruct, Llama-3.1-8B-Instruct, Seed-Coder-8B-Instruct, Yi-Coder-9B-Chat | [Paper](https://arxiv.org/abs/2508.05242) | [Code](https://github.com/sijieaaa/CodeBoost) |
| PRLCoder | Process | CodeT5+, Unixcoder, T5-base | [Paper](https://arxiv.org/abs/2502.01715) | - |
| o1-Coder | Process | DeepSeek-1.3B-Instruct | [Paper](https://arxiv.org/abs/2412.00154) | [Code](https://github.com/ADaM-BJTU/O1-CODER) |
| CodeFavor | Process | Mistral-NeMo-12B-Instruct, Gemma-2-9B-Instruct, Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.3 | [Paper](https://arxiv.org/abs/2410.03837) | [Code](https://llm-code-preference.github.io/) |
| Focused-DPO | Process | DeepSeek-Coder-6.7B-Base/Instruct, Magicoder-S-DS-6.7B, Qwen2.5-Coder-7B-Instruct | [Paper](https://aclanthology.org/2025.findings-acl.498/) | - |
| **_RL for Iterative Code Refinement_** ||||| 
| RLEF | Outcome | Llama-3.0-8B-Instruct, Llama-3.1-8B/70B-Instruct | [Paper](https://openreview.net/forum?id=PzSG5nKe1q) | - |
| μCode | Outcome | Llama-3.2-1B/8B-Instruct | [Paper](https://openreview.net/forum?id=aJeLhLcsh0) | [Code](https://github.com/portal-cornell/muCode) |
| R1-Code-Interpreter | Outcome | Qwen2.5-7B/14B-Instruct-1M, Qwen2.5-3B-Instruct | [Paper](https://arxiv.org/abs/2505.21668) | [Code](https://github.com/yongchao98/R1-Code-Interpreter) |
| IterPref | Process | Deepseek-Coder-7B-Instruct, Qwen2.5-Coder-7B, StarCoder2-15B | [Paper](https://arxiv.org/abs/2503.02783) | - |
| LeDex | Process | StarCoder-15B, CodeLlama-7B/13B | [Paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/3ea832724870c700f0a03c665572e2a9-Paper-Conference.pdf) | - |
| CTRL | Process | Qwen2.5-Coder-7B/14B/32B-Instruct | [Paper](https://openreview.net/forum?id=UVoxPlv5E1) | [Code](https://github.com/HKUNLP/critic-rl) |
| ReVeal | Process | DAPO-Qwen-32B, Qwen2.5-32B-Instruc(not-working) | [Paper](https://arxiv.org/abs/2506.11442) | - |
| Posterior-GRPO | Process | Qwen2.5-Coder-3B/7B-Base, Qwen2.5-Math-7B | [Paper](https://arxiv.org/abs/2508.05170) | - |
| Policy Filtration for RLHF | Process | DeepSeek-Coder-6.7B, Qwen1.5-7B | [Paper](https://openreview.net/forum?id=L8hYdTQVcs) | [Code](https://github.com/swtheing/PF-PPO-RLHF) |
| **_RL for Automated Software Engineering (SWE)_** ||||| 
| DeepSWE | Outcome | Qwen3-32B | [Blog](https://pretty-radio-b75.notion.site/DeepSWE-Training-a-Fully-Open-sourced-State-of-the-Art-Coding-Agent-by-Scaling-RL-22281902c1468193aabbe9a8c59bbe33) | [Code](https://github.com/agentica-project/rllm) |
| SWE-RL | Outcome | Llama-3.3-70B-Instruct | [Paper](https://arxiv.org/abs/2502.18449) | [Code](https://github.com/facebookresearch/swe-rl) |
| Satori-SWE | Outcome | Qwen-2.5-Math-7B | [Paper](https://openreview.net/forum?id=j4FXxMiDjL) | [Code](https://github.com/satori-reasoning/Satori) |
| RLCoder | Outcome | CodeLlama7B, StartCoder-7B, StarCoder2-7B, DeepSeekCoder-1B/7B | [Paper](https://www.computer.org/csdl/proceedings-article/icse/2025/056900a165/215aWzRTwjK) | [Code](https://github.com/DeepSoftwareAnalytics/RLCoder) |
| Qwen3-Coder | Outcome | - | [Paper](https://arxiv.org/pdf/2505.09388) | [Code](https://github.com/QwenLM/Qwen3) |
| ML-Agent | Outcome | Qwen2.5-7B-Base/Instruct, DeepSeek-R1-Distill-Qwen-7B | [Paper](https://arxiv.org/pdf/2505.23723) | [Code](https://github.com/MASWorks/ML-Agent) |
| Golubev et al. | Process | Qwen2.5-72B-Instruct | [Paper](https://arxiv.org/abs/2508.03501) | - |
| SWEET-RL | Process | Llama-3.1-8B/70B-Instruct | [Paper](https://arxiv.org/abs/2503.15478) | [Code](https://github.com/facebookresearch/sweet_rl) |
| DeepAnalyze | Outcome | DeepSeek-R1-Distill-Qwen3-8B | [Paper](https://arxiv.org/abs/2510.16872) | [Code](https://github.com/ruc-datalab/DeepAnalyze) |

---

## Sec4.3 Task: Mathematical Agent

| Method | Reward | Link | Resource |
|--------|--------|------|----------|
| **_RL for Informal Mathematical Reasoning_** ||||
| ARTIST | Outcome | [Paper](https://arxiv.org/abs/2505.01441) | - |
| ToRL | Outcome | [Paper](https://arxiv.org/abs/2503.05592) | [Code](https://github.com/GAIR-NLP/ToRL) [Model](https://huggingface.co/GAIR/ToRL-7B) |
| ZeroTIR | Outcome | [Paper](https://arxiv.org/abs/2505.07773) | [Code](https://github.com/yyht/openrlhf_async_pipline) [Model](https://huggingface.co/htxu91/zero-tir-7b-550step) |
| TTRL | Outcome | [Paper](https://arxiv.org/abs/2504.16084) | [Code](https://github.com/PRIME-RL/TTRL) |
| RENT | Outcome | [Paper](https://arxiv.org/abs/2505.22660) | [Code](https://github.com/satrams/rent-rl) [Website](https://rent-rl.github.io/) |
| Satori | Outcome | [Paper](https://openreview.net/forum?id=j4FXxMiDjL) | [Code](https://github.com/satori-reasoning/Satori) [Model](https://huggingface.co/Satori-reasoning) [Website](https://rent-rl.github.io/) |
| 1-shot RLVR | Outcome | [Paper](https://arxiv.org/abs/2504.20571) | [Code](https://github.com/ypwang61/One-Shot-RLVR) [Model](https://huggingface.co/collections/ypwang61/one-shot-rlvr-6827f72c3359b2ffe75fc1a8) |
| Prover-Verifier Games (legibility) | Outcome | [Paper](https://arxiv.org/abs/2407.13692) | - |
| rStar2-Agent | Outcome | [Paper](https://arxiv.org/abs/2508.20722) | [Code](https://github.com/microsoft/rStar) |
| START | Process | [Paper](https://arxiv.org/abs/2503.04625) | - |
| LADDER | Process | [Paper](https://arxiv.org/abs/2503.00735) | - |
| SWiRL | Process | [Paper](https://arxiv.org/abs/2504.04736) | - |
| RLoT | Process | [Paper](https://arxiv.org/abs/2505.14140) | [Code](https://anonymous.4open.science/r/RL-LLM-Reasoning-1A30) |
| AutoTIR | Process | [Paper](https://arxiv.org/pdf/2507.21836) | [Code](https://github.com/weiyifan1023/AutoTIR) |
| **_RL for Formal Mathematical Reasoning_** ||||
| DeepSeek-Prover-v1.5 | Outcome | [Paper](https://openreview.net/forum?id=I4YAIwrsXa) | [Code](https://github.com/deepseek-ai/DeepSeek-Prover-V1.5) [Model](https://huggingface.co/deepseek-ai) |
| Leanabell-Prover | Outcome | [Paper](https://arxiv.org/abs/2504.06122) | [Code](https://github.com/Leanabell-LM/Leanabell-Prover) [Model](https://huggingface.co/collections/stoney0062/leanabell-prover-67fe4fae1dcf1d7221e309e9) |
| Kimina-Prover (Preview) | Outcome | [Paper](https://arxiv.org/abs/2504.11354) | [Code](https://github.com/MoonshotAI/Kimina-Prover-Preview) [Model](https://huggingface.co/collections/AI-MO/kimina-prover-preview-67fb536b883d60e7ca25d7f9) |
| Seed-Prover | Outcome | [Paper](https://arxiv.org/abs/2507.23726) | [Code](https://github.com/ByteDance-Seed/Seed-Prover) |
| DeepSeek-Prover-v2 | Process | [Paper](doi.org/10.48550/arXiv.2405.04434) | [Code](https://github.com/deepseek-ai/DeepSeek-V2) [Model](https://huggingface.co/deepseek-ai) |
| ProofNet++ | Process | [Paper](https://arxiv.org/abs/2505.24230) | - |
| Leanabell-Prover-v2 | Process | [Paper](https://arxiv.org/abs/2507.08649) | [Code](https://github.com/Leanabell-LM/Leanabell-Prover-V2) |
| **_Hybrid_** ||||
| InternLM2.5-StepProver | Hybrid | [Paper](https://openreview.net/forum?id=qwCqeIg5iI) | [Code](https://github.com/InternLM/InternLM-Math) |
| Lean-STaR | Hybrid | [Paper](https://openreview.net/forum?id=SOWZ59UyNc) | [Code](https://github.com/Lagooon/LeanSTaR) [Model](https://huggingface.co/ScalableMath/Lean-STaR-plus) [Website](https://leanstar.github.io/) |
| STP | Hybrid | [Paper](https://openreview.net/forum?id=zWArMedNuW) | [Code](https://github.com/kfdong/STP) [Model](https://huggingface.co/kfdong/STP_model_Lean_0320) |


---

## Sec4.4 Task: GUI Agent

| Method | Paradigm | Environment | Link | Resource |
|--------|----------|-------------|------|----------|
| **_Non-RL GUI Agents_** |||||
| MM-Navigator | Vanilla VLM | - | [Paper](https://arxiv.org/abs/2311.07562) | [Code](https://github.com/zzxslp/MM-Navigator) |
| SeeAct | Vanilla VLM | - | [Paper](https://proceedings.mlr.press/v235/zheng24e.html) | [Code](https://github.com/OSU-NLP-Group/SeeAct) |
| TRISHUL | Vanilla VLM | - | [Paper](https://arxiv.org/abs/2502.08226) | - |
| InfiGUIAgent | SFT | - | [Paper](https://openreview.net/forum?id=p0h9XJ7fMH) | [Code](https://github.com/InfiXAI/InfiGUIAgent) [Model](https://huggingface.co/datasets/rootsautomation/ScreenSpot) [Website](https://b7277.github.io/InfiGUIAgent.github.io/) |
| UI-AGILE | SFT | - | [Paper](https://arxiv.org/abs/2507.22025) | [Code](https://github.com/KDEGroup/UI-AGILE) [Model](https://huggingface.co/KDEGroup/UI-AGILE) |
| TongUI | SFT | - | [Paper](https://arxiv.org/abs/2504.12679) | [Code](https://github.com/TongUI-agent/TongUI-agent) [Model](https://huggingface.co/collections/Bofeee5675/tongui-67f611e2d48b2b6e0d2ba3ee) [Website](https://tongui-agent.github.io/) |
| **_RL-based GUI Agents_** |||||
| GUI-R1 | RL | Static | [Paper](https://arxiv.org/pdf/2504.10458) | [Code](https://github.com/ritzz-ai/GUI-R1) [Model](https://huggingface.co/ritzzai/GUI-R1) |
| UI-R1 | RL | Static | [Paper](https://arxiv.org/abs/2503.21620) | [Code](https://github.com/lll6gg/UI-R1) [Model](https://huggingface.co/LZXzju/Qwen2.5-VL-3B-UI-R1-E) |
| InFiGUI-R1 | RL | Static | [Paper](https://arxiv.org/abs/2504.14239) | [Code](https://github.com/InfiXAI/InfiGUI-R1) [Model](https://huggingface.co/InfiX-ai/InfiGUI-R1-3B) |
| AgentCPM | RL | Static | [Paper](https://arxiv.org/abs/2506.01391) | [Code](https://github.com/OpenBMB/AgentCPM-GUI) [Model](https://huggingface.co/openbmb/AgentCPM-GUI) |
| WebAgent-R1 | RL | Interactive | [Paper](https://openreview.net/forum?id=KqrYTALRjH) | - |
| Vattikonda et al. | RL | Interactive | [Paper](https://arxiv.org/abs/2507.04103) | - |
| UI-TARS | RL | Interactive | [Paper](https://arxiv.org/abs/2501.12326) | [Code](https://github.com/bytedance/UI-TARS) [Model](https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B) [Website](https://seed-tars.com/) |
| DiGiRL | RL | Interactive | [Paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/1704ddd0bb89f159dfe609b32c889995-Paper-Conference.pdf) | [Code](https://github.com/DigiRL-agent/digirl) [Model](https://huggingface.co/collections/JackBAI/digirl-6682ea42bdfb5af9bfc5f29f) [Website](https://digirl-agent.github.io/) |
| ZeroGUI | RL | Interactive | [Paper](https://arxiv.org/pdf/2505.23762) | [Code](https://github.com/OpenGVLab/ZeroGUI) |
| MobileGUI-RL | RL | Interactive | [Paper](https://arxiv.org/abs/2507.05720) | - |


---

## Sec4.5 Task: RL in Vision Agents
TO BE ADDED

---

## Sec4.6 Task: RL in Embodied Agents
TO BE ADDED

---

## Sec4.7 Task: RL in Multi-Agent Systems

“Dynamic” denotes whether the multi-agent system is task-dynamic, i.e., processes different task queries with different configurations (agent count, topologies, reasoning depth, prompts, etc).  
“Train” denotes whether the method involves training the LLM backbone of agents.  

| Method | Dynamic | Train | RL Algorithm | Link | Resource |
|--------|----------|-------|--------------|------|----------|
| **_RL-Free Multi-Agent Systems (not exhaustive)_** |||||  
| CAMEL | ✗ | ✗ | - | [Paper](https://dl.acm.org/doi/10.5555/3666122.3668386) | [Code](https://github.com/camel-ai/camel) [Model](https://huggingface.co/camel-ai) |
| MetaGPT | ✗ | ✗ | - | [Paper](https://openreview.net/forum?id=VtmBAGCN7o) | [Code](https://github.com/FoundationAgents/MetaGPT) |
| MAD | ✗ | ✗ | - | [Paper](https://aclanthology.org/2024.emnlp-main.992/) | [Code](https://github.com/Skytliang/Multi-Agents-Debate) |
| MoA | ✗ | ✗ | - | [Paper](https://openreview.net/forum?id=h0ZfDIrj7T) | [Code](https://github.com/togethercomputer/moa) |
| AFlow | ✗ | ✗ | - | [Paper](https://openreview.net/forum?id=z5uVAKwmjf) | [Code](https://github.com/FoundationAgents/AFlow) |
| **_RL-Based Multi-Agent Training_** |||||  
| GPTSwarm | ✗ | ✗ | policy gradient | [Paper](https://openreview.net/forum?id=uTC9AFXIhg) | [Code](https://github.com/metauto-ai/gptswarm) [Website](https://gptswarm.org/) |
| MaAS | ✓ | ✗ | policy gradient | [Paper](https://openreview.net/forum?id=imcyVlzpXh) | [Code](https://github.com/bingreeky/MaAS) |
| G-Designer | ✓ | ✗ | policy gradient | [Paper](https://openreview.net/forum?id=LpE54NUnmO) | [Code](https://github.com/yanweiyue/GDesigner) |
| MALT | ✗ | ✓ | DPO | [Paper](https://openreview.net/forum?id=lIf7grAC7n) | - |
| MARFT | ✗ | ✓ | MARFT | [Paper](https://arxiv.org/abs/2504.16129) | [Code](https://github.com/jwliao-ai/MARFT) |
| MAPoRL | ✓ | ✓ | PPO | [Paper](https://aclanthology.org/2025.acl-long.1459/) | [Code](https://github.com/chanwoo-park-official/MAPoRL) |
| MLPO | ✓ | ✓ | MLPO | [Paper](https://arxiv.org/abs/2507.08960) | - |
| ReMA | ✓ | ✓ | MAMRP | [Paper](https://arxiv.org/abs/2503.09501) | [Code](https://github.com/ziyuwan/ReMA-public) |
| FlowReasoner | ✓ | ✓ | GRPO | [Paper](https://arxiv.org/abs/2504.15257) | [Code](https://github.com/sail-sg/FlowReasoner) |
| LERO | ✓ | ✓ | MLPO | [Paper](https://arxiv.org/abs/2503.21807) | - |
| CURE | ✗ | ✓ | rule-based RL | [Paper](https://arxiv.org/abs/2506.03136) | [Code](https://github.com/Gen-Verse/CURE) [Model](https://huggingface.co/collections/Gen-Verse/reasonflux-coder-6833109ed9300c62deb32c6b) |
| MMedAgent-RL | ✗ | ✓ | GRPO | [Paper](https://arxiv.org/abs/2506.00555) | - |
| OWL |  ✓ | ✓ | DPO | [Paper](https://arxiv.org/abs/2505.23885) | [Code](https://github.com/camel-ai/owl) |

## Sec4.8. Task: Other Tasks
TO BE ADDED

## Sec5.1 Environments

The agent capabilities are denoted by:  
① Reasoning, ② Planning, ③ Tool Use, ④ Memory, ⑤ Collaboration, ⑥ Self-Improve.  

| Environment / Benchmark | Agent Capability | Task Domain | Modality | Link | Resource |
|--------------------------|------------------|-------------|----------|------|----------|
| LMRL-Gym | ①, ④ | Interaction | Text | [Paper](https://openreview.net/forum?id=hmGhP5DO2W) | [Code](https://github.com/abdulhaim/LMRL-Gym) |
| ALFWorld | ②, ① | Embodied, Text Games | Text | [Paper](https://openreview.net/forum?id=0IOX0YcCdTn) | [Code](https://github.com/alfworld/alfworld) [Website](https://alfworld.github.io/) |
| TextWorld | ②, ① | Text Games | Text | [Paper](https://arxiv.org/pdf/1806.11532) | [Code](https://github.com/microsoft/TextWorld) |
| ScienceWorld | ①, ② | Embodied, Science | Text | [Paper](https://aclanthology.org/2022.emnlp-main.775/) | [Code](https://github.com/allenai/ScienceWorld) [Website](https://sciworld.apps.allenai.org/) |
| AgentGym | ①, ④ | Text Games | Text | [Paper](https://aclanthology.org/2025.acl-long.1355/) | [Code](https://github.com/WooooDyy/AgentGym) [Website](https://agentgym.github.io/) |
| Agentbench | ① | General | Text, Visual | [Paper](https://openreview.net/forum?id=zAdUB0aCTQ) | [Code](https://github.com/THUDM/AgentBench) |
| InternBootcamp | ① | General, Coding, Logic | Text | [Paper](https://arxiv.org/abs/2508.08636) | [Code](https://github.com/InternLM/InternBootcamp) |
| LoCoMo | ④ | Interaction | Text | [Paper](https://arxiv.org/abs/2402.17753) | [Code](https://github.com/snap-research/LoCoMo) [Website](https://snap-research.github.io/locomo/) |
| MemoryAgentBench | ④ | Interaction | Text | [Paper](https://arxiv.org/abs/2507.05257) | [Code](https://github.com/HUST-AI-HYZ/MemoryAgentBench) |
| WebShop | ②, ③ | Web | Text | [Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/82ad13ec01f9fe44c01cb91814fd7b8c-Paper-Conference.pdf) | [Code](https://github.com/princeton-nlp/WebShop) [Website](https://webshop-pnlp.github.io/) |
| Mind2Web | ②, ③ | Web | Text, Visual | [Paper](https://arxiv.org/abs/2506.21506) | [Code](https://github.com/OSU-NLP-Group/Mind2Web-2) [Website](https://osu-nlp-group.github.io/Mind2Web-2/) |
| WebArena | ②, ③ | Web | Text | [Paper](https://openreview.net/forum?id=oKn9c6ytLx) | [Code](https://github.com/web-arena-x/webarena) [Website](https://webarena.dev/) |
| VisualwebArena | ①, ②, ③ | Web | Text, Visual | [Paper](https://arxiv.org/abs/2401.13649) | [Code](https://github.com/web-arena-x/visualwebarena) [Website](https://jykoh.com/vwa) |
| AppWorld | ②, ③ | App | Text | [Paper](https://aclanthology.org/2024.acl-long.850/) | [Code](https://github.com/stonybrooknlp/appworld) [Website](https://appworld.dev/) |
| AndroidWorld | ②, ③ | GUI, App | Text, Visual | [Paper](https://openreview.net/forum?id=il5yUQsrjC) | [Code](https://github.com/google-research/android_world) |
| OSWorld | ②, ③ | GUI, OS | Text, Visual | [Paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/5d413e48f84dc61244b6be550f1cd8f5-Paper-Datasets_and_Benchmarks_Track.pdf) | [Code](https://github.com/xlang-ai/OSWorld) [Website](https://os-world.github.io/) |
| Debug-Gym | ①, ③ | SWE | Text | [Paper](https://arxiv.org/abs/2503.21557) | [Code](https://github.com/microsoft/debug-gym) [Website](https://microsoft.github.io/debug-gym/) |
| MLE-Dojo | ②, ① | MLE | Text | [Paper](https://arxiv.org/abs/2505.07782) | [Code](https://github.com/MLE-Dojo/MLE-Dojo) [Website](https://mle-dojo.github.io/MLE-Dojo-page/) |
| τ-bench | ①, ③ | SWE | Text | [Paper](https://arxiv.org/abs/2506.07982) | [Code](https://github.com/sierra-research/tau2-bench) |
| TheAgentCompany | ②, ③, ⑤ | SWE | Text | [Paper](https://arxiv.org/abs/2412.14161) | [Code](https://github.com/TheAgentCompany/TheAgentCompany) [Website](https://the-agent-company.com/) |
| MedAgentGym | ① | Science | Text | [Paper](https://arxiv.org/abs/2506.04405) | [Code](https://github.com/wshi83/MedAgentGym) |
| SecRepoBench | ①, ③ | Coding, Security | Text | [Paper](https://arxiv.org/abs/2504.21205) | - |
| R2E-Gym | ①, ② | SWE | Text | [Paper](https://openreview.net/forum?id=7evvwwdo3z) | [Code](https://github.com/R2E-Gym/R2E-Gym) [Website](https://r2e-gym.github.io/) |
| HumanEval | ① | Coding | Text | [Paper](https://arxiv.org/abs/2107.03374) | [Code](https://github.com/openai/human-eval) |
| MBPP | ① | Coding | Text | [Paper](https://arxiv.org/abs/2108.07732) | [Code](https://github.com/google-research/google-research/tree/master/mbpp) |
| BigCodeBench | ① | Coding | Text | [Paper](https://openreview.net/forum?id=YrycTjllL0) | [Code](https://github.com/bigcode-project/bigcodebench) [Website](https://bigcode-bench.github.io/) |
| LiveCodeBench | ① | Coding | Text | [Paper](https://openreview.net/forum?id=chfJJYC3iL) | [Code](https://github.com/LiveCodeBench/LiveCodeBench) [Website](https://livecodebench.github.io) |
| SWE-bench | ①, ③ | SWE | Text | [Paper](https://openreview.net/forum?id=VTF8yNQM66) | [Code](https://github.com/swe-bench/SWE-bench) [Website](https://www.swebench.com/) |
| SWE-rebench | ①, ③ | SWE | Text | [Paper](https://arxiv.org/abs/2505.20411) | [Website](https://swe-rebench.com/) |
| DevBench | ②, ① | SWE | Text | [Paper](https://aclanthology.org/2025.coling-main.502/) | [Code](https://github.com/open-compass/DevEval) |
| ProjectEval | ②, ① | SWE | Text | [Paper](https://aclanthology.org/2025.findings-acl.1036/) | [Code](https://github.com/RyanLoil/ProjectEval/) [Website](https://ryanloil.github.io/ProjectEval/) |
| DA-Code | ①, ③ | Data Science, SWE | Text | [Paper](https://aclanthology.org/2024.emnlp-main.748/) | [Code](https://aclanthology.org/2024.emnlp-main.748/) [Website](https://github.com/yiyihum/da-code) |
| ColBench | ②, ①, ③ | SWE, Web Dev | Text | [Paper](https://arxiv.org/abs/2503.15478) | [Code](https://arxiv.org/abs/2503.15478) [Website](https://github.com/facebookresearch/sweet_rl) |
| NoCode-bench | ②, ① | SWE | Text | [Paper](https://arxiv.org/abs/2507.18130) | [Code](https://github.com/NoCode-bench/NoCode-bench) [Website](https://nocodebench.org/) |
| MLE-Bench | ②, ①, ③ | MLE | Text | [Paper](https://openreview.net/forum?id=6s5uXNWGIh) | [Code](https://github.com/openai/mle-bench/) [Website](https://openai.com/index/mle-bench/) |
| PaperBench | ②, ①, ③ | MLE | Text | [Paper](https://openreview.net/forum?id=xF5PuTLPbn) | [Code](https://github.com/openai/preparedness/tree/main/project/paperbench) [Website](https://openai.com/index/paperbench/) |
| Crafter | ②, ④ | Game | Visual | [Paper](https://openreview.net/forum?id=1W0z96MFEoH) | [Code](https://openreview.net/forum?id=1W0z96MFEoH) [Website](https://danijar.com/crafter) |
| Craftax | ②, ④ | Game | Visual | [Paper](https://openreview.net/forum?id=hg4wXlrQCV) | [Code](https://github.com/MichaelTMatthews/Craftax) |
| ELLM (Crafter variant) | ②, ① | Game | Visual | [Paper](https://proceedings.mlr.press/v202/du23f.html) | [Code](https://proceedings.mlr.press/v202/du23f.html) [Website](https://github.com/yuqingd/ellm) |
| SMAC / SMAC-Exp | ⑤, ② | Game | Visual | [Paper](https://arxiv.org/abs/1902.04043) | [Code](https://github.com/oxwhirl/smac) |
| Factorio | ②, ① | Game | Visual | [Paper](https://arxiv.org/abs/2503.09617) | [Code](https://github.com/JackHopkins/factorio-learning-environment) [Website](https://jackhopkins.github.io/factorio-learning-environment/) |

## Sec5.2 Frameworks

| Framework | Type | Key Features | Link | Resource |
|-----------|------|--------------|------|----------|
| **_Agentic RL Frameworks_** |||||
| Verifiers | Agent RL / LLM RL | Verifiable environment setup | - | [Code](https://github.com/willccbb/verifiers) |
| SkyRL-v0/v0.1 | Agent RL | Long-horizon real-world training | [Blog (v0)](https://novasky-ai.notion.site/skyrl-v0) [Blog (v0.1)](https://novasky-ai.notion.site/skyrl-v01) | [Code](https://github.com/NovaSky-AI/SkyRL) |
| AREAL | Agent RL / LLM RL | Asynchronous training | [Paper](https://openreview.net/forum?id=qJ0okaW9Z9) | [Code](https://github.com/inclusionAI/AReaL) |
| MARTI | Multi-agent RL / LLM RL | Integrated multi-agent training | - | [Code](https://github.com/TsinghuaC3I/MARTI) |
| EasyR1 | Agent RL / LLM RL | Multimodal support | - | [Code](https://github.com/hiyouga/EasyR1) |
| AgentFly | Agent RL | Scalable asynchronous execution | [Paper](https://arxiv.org/abs/2507.14897) | [Code](https://github.com/Agent-One-Lab/AgentFly) |
| Agent Lightning | Agent RL | Decoupled hierarchical RL | [Paper](https://arxiv.org/abs/2508.03680) | [Code](https://github.com/microsoft/agent-lightning) |
| **_RLHF and LLM Fine-tuning Frameworks_** |||||
| OpenRLHF | RLHF / LLM RL | High-performance scalable RLHF | [Paper](https://arxiv.org/abs/2405.11143) | [Code](https://github.com/OpenRLHF/OpenRLHF) |
| TRL | RLHF / LLM RL | Hugging Face RLHF | - | [Code](https://github.com/huggingface/trl) |
| trlX | RLHF / LLM RL | Distributed large-model RLHF | [Paper](https://aclanthology.org/2023.emnlp-main.530) | [Code](https://github.com/CarperAI/trlx) |
| HybridFlow | RLHF / LLM RL | Streamlined experiment management | [Paper](http://dx.doi.org/10.1145/3689031.3696075) | [Code](https://github.com/volcengine/verl) |
| SLiMe | RLHF / LLM RL | High-performance async RL | - | [Code](https://github.com/THUDM/slime) |
| **_General-purpose RL Frameworks_** |||||
| RLlib | General RL / Multi-agent RL | Production-grade scalable library | [Paper](https://proceedings.mlr.press/v80/liang18b.html) | [Code](https://github.com/ray-project/ray/tree/master/rllib) |
| Acme | General RL | Modular distributed components | [Paper](https://arxiv.org/abs/2006.00979) | [Code](https://github.com/google-deepmind/acme) |
| Tianshou | General RL | High-performance PyTorch platform | [Paper](https://jmlr.org/papers/v22/20-1364.html) | [Code](https://github.com/thu-ml/tianshou/) |
| Stable Baselines3 | General RL | Reliable PyTorch algorithms | [Paper](https://jmlr.org/papers/v22/20-1364.html) | [Code](https://github.com/DLR-RM/stable-baselines3) |
| PFRL | General RL | Benchmarked prototyping algorithms | [Paper](https://jmlr.org/papers/v22/20-376.html) | [Code](https://github.com/pfnet/pfrl) |

