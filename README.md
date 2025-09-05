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
| **RL for Code Generation** ||||| 
| AceCoder | Outcome | Qwen2.5-Coder-7B-Base/Instruct, Qwen2.5-7B-Instruct | [Paper](https://arxiv.org/abs/2502.01718) | [Code](https://github.com/TIGER-AI-Lab/AceCoder) |
| DeepCoder-14B | Outcome | Deepseek-R1-Distilled-Qwen-14B | [Blog](https://pretty-radio-b75.notion.site/DeepCoder-A-Fully-Open-Source-14B-Coder-at-O3-mini-Level-1cf81902c14680b3bee5eb349a512a51) | [Code](https://github.com/agentica-project/rllm) |
| RLTF | Outcome | CodeGen-NL 2.7B, CodeT5 | [Paper](https://openreview.net/forum?id=hjYmsV6nXZ) | [Code](https://github.com/Zyq-scut/RLTF) |
| CURE | Outcome | Qwen2.5-7B/14B-Instruct, Qwen3-4B | [Paper](https://arxiv.org/abs/2506.03136) | [Code](https://github.com/Gen-Verse/CURE) |
| Absolute Zero | Outcome | Qwen2.5-7B/14B, Qwen2.5-Coder-3B/7B/14B, Llama-3.1-8B | [Paper](https://arxiv.org/abs/2505.03335) | [Code](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner) |
| StepCoder | Process | DeepSeek-Coder-Instruct-6.7B | [Paper](https://aclanthology.org/2024.acl-long.251/) | [Code](https://github.com/Ablustrund/APPS_Plus) |
| Process Supervision-Guided PO | Process | - | [Paper](https://openreview.net/forum?id=Cn5Z0MUPZT) | - |
| CodeBoost | Process | Qwen2.5-Coder-7B-Instruct, Llama-3.1-8B-Instruct, Seed-Coder-8B-Instruct, Yi-Coder-9B-Chat | [Paper](https://arxiv.org/abs/2508.05242) | [Code](https://github.com/sijieaaa/CodeBoost) |
| PRLCoder | Process | CodeT5+, Unixcoder, T5-base | [Paper](https://arxiv.org/abs/2502.01715) | - |
| o1-Coder | Process | DeepSeek-1.3B-Instruct | [Paper](https://arxiv.org/abs/2412.00154) | [Code](https://github.com/ADaM-BJTU/O1-CODER) |
| CodeFavor | Process | Mistral-NeMo-12B-Instruct, Gemma-2-9B-Instruct, Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.3 | [Paper](https://arxiv.org/abs/2410.03837) | [Code](https://llm-code-preference.github.io/) |
| Focused-DPO | Process | DeepSeek-Coder-6.7B-Base/Instruct, Magicoder-S-DS-6.7B, Qwen2.5-Coder-7B-Instruct | [Paper](https://aclanthology.org/2025.findings-acl.498/) | - |
| **RL for Iterative Code Refinement** ||||| 
| RLEF | Outcome | Llama-3.0-8B-Instruct, Llama-3.1-8B/70B-Instruct | [Paper](https://openreview.net/forum?id=PzSG5nKe1q) | - |
| μCode | Outcome | Llama-3.2-1B/8B-Instruct | [Paper](https://openreview.net/forum?id=aJeLhLcsh0) | [Code](https://github.com/portal-cornell/muCode) |
| R1-Code-Interpreter | Outcome | Qwen2.5-7B/14B-Instruct-1M, Qwen2.5-3B-Instruct | [Paper](https://arxiv.org/abs/2505.21668) | [Code](https://github.com/yongchao98/R1-Code-Interpreter) |
| IterPref | Process | Deepseek-Coder-7B-Instruct, Qwen2.5-Coder-7B, StarCoder2-15B | [Paper](https://arxiv.org/abs/2503.02783) | - |
| LeDex | Process | StarCoder-15B, CodeLlama-7B/13B | [Paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/3ea832724870c700f0a03c665572e2a9-Paper-Conference.pdf) | - |
| CTRL | Process | Qwen2.5-Coder-7B/14B/32B-Instruct | [Paper](https://openreview.net/forum?id=UVoxPlv5E1) | [Code](https://github.com/HKUNLP/critic-rl) |
| ReVeal | Process | DAPO-Qwen-32B, Qwen2.5-32B-Instruc(not-working) | [Paper](https://arxiv.org/abs/2506.11442) | - |
| Posterior-GRPO | Process | Qwen2.5-Coder-3B/7B-Base, Qwen2.5-Math-7B | [Paper](https://arxiv.org/abs/2508.05170) | - |
| Policy Filtration for RLHF | Process | DeepSeek-Coder-6.7B, Qwen1.5-7B | [Paper](https://openreview.net/forum?id=L8hYdTQVcs) | [Code](https://github.com/swtheing/PF-PPO-RLHF) |
| **RL for Automated Software Engineering (SWE)** ||||| 
| DeepSWE | Outcome | Qwen3-32B | [Blog](https://pretty-radio-b75.notion.site/DeepSWE-Training-a-Fully-Open-sourced-State-of-the-Art-Coding-Agent-by-Scaling-RL-22281902c1468193aabbe9a8c59bbe33) | [Code](https://github.com/agentica-project/rllm) |
| SWE-RL | Outcome | Llama-3.3-70B-Instruct | [Paper](https://arxiv.org/abs/2502.18449) | [Code](https://github.com/facebookresearch/swe-rl) |
| Satori-SWE | Outcome | Qwen-2.5-Math-7B | [Paper](https://openreview.net/forum?id=j4FXxMiDjL) | [Code](https://github.com/satori-reasoning/Satori) |
| SWEET-RL | Outcome | Llama-3.1-8B/70B-Instruct | [Paper](https://arxiv.org/abs/2503.15478) | [Code](https://github.com/facebookresearch/sweet_rl) |
| RLCoder | Outcome | CodeLlama7B, StartCoder-7B, StarCoder2-7B, DeepSeekCoder-1B/7B | [Paper](https://www.computer.org/csdl/proceedings-article/icse/2025/056900a165/215aWzRTwjK) | [Code](https://github.com/DeepSoftwareAnalytics/RLCoder) |
| Qwen3-Coder | Outcome | - | [Paper](https://arxiv.org/pdf/2505.09388) | [Code](https://github.com/QwenLM/Qwen3) |
| ML-Agent | Outcome | Qwen2.5-7B-Base/Instruct, DeepSeek-R1-Distill-Qwen-7B | [Paper](https://arxiv.org/pdf/2505.23723) | [Code](https://github.com/MASWorks/ML-Agent) |
| Golubev et al. | Process | Qwen2.5-72B-Instruct | [Paper](https://arxiv.org/abs/2508.03501) | - |

---

## Sec4.3 Task: Mathematical Agent
TO BE ADDED

---

## Sec4.4 Task: GUI Agent
# A summary of methods for GUI agents, categorized by training paradigm and environment complexity

| Method | Paradigm | Environment | Link | Resource |
|--------|----------|-------------|------|----------|
| **Non-RL GUI Agents** |||||
| MM-Navigator | Vanilla VLM | - | [Paper](https://arxiv.org/abs/2311.07562) | [Code](https://github.com/zzxslp/MM-Navigator) |
| SeeAct | Vanilla VLM | - | [Paper](https://proceedings.mlr.press/v235/zheng24e.html) | [Code](https://github.com/OSU-NLP-Group/SeeAct) |
| TRISHUL | Vanilla VLM | - | [Paper](https://arxiv.org/abs/2502.08226) | - |
| InfiGUIAgent | SFT | - | [Paper](https://openreview.net/forum?id=p0h9XJ7fMH) | [Code](https://github.com/InfiXAI/InfiGUIAgent) [Model](https://huggingface.co/datasets/rootsautomation/ScreenSpot) [Website](https://b7277.github.io/InfiGUIAgent.github.io/) |
| UI-AGILE | SFT | - | [Paper](https://arxiv.org/abs/2507.22025) | [Code](https://github.com/KDEGroup/UI-AGILE) [Model](https://huggingface.co/KDEGroup/UI-AGILE) |
| TongUI | SFT | - | [Paper](https://arxiv.org/abs/2504.12679) | [Code](https://github.com/TongUI-agent/TongUI-agent) [Model](https://huggingface.co/collections/Bofeee5675/tongui-67f611e2d48b2b6e0d2ba3ee) [Website](https://tongui-agent.github.io/) |
| **RL-based GUI Agents** |||||
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
| Method | Paradigm | Link | Resource |
|--------|----------|------|----------|
| **RL-Free Multi-Agent Systems (not exhaustive)** ||||
| CAMEL | - | [Paper](https://dl.acm.org/doi/10.5555/3666122.3668386) | [Code](https://github.com/camel-ai/camel) [Model](https://huggingface.co/camel-ai) |
| MetaGPT | No-train | [Paper](https://openreview.net/forum?id=VtmBAGCN7o) | [Code](https://github.com/FoundationAgents/MetaGPT) |
| MAD | No-train | [Paper](https://aclanthology.org/2024.emnlp-main.992/) | [Code](https://github.com/Skytliang/Multi-Agents-Debate) |
| MoA | No-train | [Paper](https://openreview.net/forum?id=h0ZfDIrj7T) | [Code](https://github.com/togethercomputer/moa) |
| AFlow | No-train | [Paper](https://openreview.net/forum?id=z5uVAKwmjf) | [Code](https://github.com/FoundationAgents/AFlow) |
| **RL-Based Multi-Agent Training** ||||
| MaAS | - | [Paper](https://openreview.net/forum?id=imcyVlzpXh) | [Code](https://github.com/bingreeky/MaAS) |
| G-Designer | - | [Paper](https://openreview.net/forum?id=LpE54NUnmO) | [Code](https://github.com/yanweiyue/GDesigner) |
| MALT | Off-policy (DPO) | [Paper](https://openreview.net/forum?id=lIf7grAC7n) | - |
| MARFT | On-policy | [Paper](https://arxiv.org/abs/2504.16129) | [Code](https://github.com/jwliao-ai/MARFT) |
| MAPoRL | On-policy | [Paper](https://aclanthology.org/2025.acl-long.1459/) | [Code](https://github.com/chanwoo-park-official/MAPoRL) |
| MLPO | On-policy | [Paper](https://arxiv.org/abs/2507.08960) | - |
| ReMA | On-policy | [Paper](https://arxiv.org/abs/2503.09501) | [Code](https://github.com/ziyuwan/ReMA-public) |
| FlowReasoner | On-policy | [Paper](https://arxiv.org/abs/2504.15257) | [Code](https://github.com/sail-sg/FlowReasoner) |
| LERO | On-policy | [Paper](https://arxiv.org/abs/2503.21807) | - |
| CURE | On-policy | [Paper](https://arxiv.org/abs/2506.03136) | [Code](https://github.com/Gen-Verse/CURE) [Model](https://huggingface.co/collections/Gen-Verse/reasonflux-coder-6833109ed9300c62deb32c6b) |
| MMedAgent-RL | On-policy | [Paper](https://arxiv.org/abs/2506.00555) | - |

## Sec4.8. Task: Other Tasks

“Dynamic” denotes whether the multi-agent system is task-dynamic, i.e., processes different task queries with different configurations (agent count, topologies, reasoning depth, prompts, etc).  
“Train” denotes whether the method involves training the LLM backbone of agents.  

| Method | Dynamic | Train | RL Algorithm | Link | Resource |
|--------|----------|-------|--------------|------|----------|
| **RL-Free Multi-Agent Systems (not exhaustive)** |||||  
| CAMEL | ✗ | ✗ | - | [PDF](https://dl.acm.org/doi/10.5555/3666122.3668386) | [Code](https://github.com/camel-ai/camel) [Model](https://huggingface.co/camel-ai) |
| MetaGPT | ✗ | ✗ | - | [PDF](https://openreview.net/forum?id=VtmBAGCN7o) | [Code](https://github.com/FoundationAgents/MetaGPT) |
| MAD | ✗ | ✗ | - | [PDF](https://aclanthology.org/2024.emnlp-main.992/) | [Code](https://github.com/Skytliang/Multi-Agents-Debate) |
| MoA | ✗ | ✗ | - | [PDF](https://openreview.net/forum?id=h0ZfDIrj7T) | [Code](https://github.com/togethercomputer/moa) |
| AFlow | ✗ | ✗ | - | [PDF](https://openreview.net/forum?id=z5uVAKwmjf) | [Code](https://github.com/FoundationAgents/AFlow) |
| **RL-Based Multi-Agent Training** |||||  
| GPTSwarm | ✗ | ✗ | policy gradient | [PDF](https://openreview.net/forum?id=uTC9AFXIhg) | [Code](https://github.com/metauto-ai/gptswarm) [Website](https://gptswarm.org/) |
| MaAS | ✓ | ✗ | policy gradient | [PDF](https://openreview.net/forum?id=imcyVlzpXh) | [Code](https://github.com/bingreeky/MaAS) |
| G-Designer | ✓ | ✗ | policy gradient | [PDF](https://openreview.net/forum?id=LpE54NUnmO) | [Code](https://github.com/yanweiyue/GDesigner) |
| MALT | ✗ | ✓ | DPO | [PDF](https://openreview.net/forum?id=lIf7grAC7n) | - |
| MARFT | ✗ | ✓ | MARFT | [PDF](https://arxiv.org/abs/2504.16129) | [Code](https://github.com/jwliao-ai/MARFT) |
| MAPoRL | ✓ | ✓ | PPO | [PDF](https://aclanthology.org/2025.acl-long.1459/) | [Code](https://github.com/chanwoo-park-official/MAPoRL) |
| MLPO | ✓ | ✓ | MLPO | [PDF](https://arxiv.org/abs/2507.08960) | - |
| ReMA | ✓ | ✓ | MAMRP | [PDF](https://arxiv.org/abs/2503.09501) | [Code](https://github.com/ziyuwan/ReMA-public) |
| FlowReasoner | ✓ | ✓ | GRPO | [PDF](https://arxiv.org/abs/2504.15257) | [Code](https://github.com/sail-sg/FlowReasoner) |
| LERO | ✓ | ✓ | MLPO | [PDF](https://arxiv.org/abs/2503.21807) | - |
| CURE | ✗ | ✓ | rule-based RL | [PDF](https://arxiv.org/abs/2506.03136) | [Code](https://github.com/Gen-Verse/CURE) [Model](https://huggingface.co/collections/Gen-Verse/reasonflux-coder-6833109ed9300c62deb32c6b) |
| MMedAgent-RL | ✗ | ✓ | GRPO | [PDF](https://arxiv.org/abs/2506.00555) | - |

## Sec5.1 Environments

The agent capabilities are denoted by:  
① Reasoning, ② Planning, ③ Tool Use, ④ Memory, ⑤ Collaboration, ⑥ Self-Improve.  

| Environment / Benchmark | Agent Capability | Task Domain | Modality | Link | Resource |
|--------------------------|------------------|-------------|----------|------|----------|
| LMRL-Gym | ①, ④ | Interaction | Text | [PDF](https://openreview.net/forum?id=hmGhP5DO2W) | [Code](https://github.com/abdulhaim/LMRL-Gym) |
| ALFWorld | ②, ① | Embodied, Text Games | Text | [PDF](https://openreview.net/forum?id=0IOX0YcCdTn) | [Code](https://github.com/alfworld/alfworld) [Website](https://alfworld.github.io/) |
| TextWorld | ②, ① | Text Games | Text | [PDF](https://arxiv.org/pdf/1806.11532) | [Code](https://github.com/microsoft/TextWorld) |
| ScienceWorld | ①, ② | Embodied, Science | Text | [PDF](https://aclanthology.org/2022.emnlp-main.775/) | [Code](https://github.com/allenai/ScienceWorld) [Website](https://sciworld.apps.allenai.org/) |
| AgentGym | ①, ④ | Text Games | Text | [PDF](https://aclanthology.org/2025.acl-long.1355/) | [Code](https://github.com/WooooDyy/AgentGym) [Website](https://agentgym.github.io/) |
| Agentbench | ① | General | Text, Visual | [PDF](https://openreview.net/forum?id=zAdUB0aCTQ) | [Code](https://github.com/THUDM/AgentBench) |
| InternBootcamp | ① | General, Coding, Logic | Text | [PDF](https://arxiv.org/abs/2508.08636) | [Code](https://github.com/InternLM/InternBootcamp) |
| LoCoMo | ④ | Interaction | Text | [PDF](https://arxiv.org/abs/2402.17753) | [Code](https://github.com/snap-research/LoCoMo) [Website](https://snap-research.github.io/locomo/) |
| MemoryAgentBench | ④ | Interaction | Text | [PDF](https://arxiv.org/abs/2507.05257) | [Code](https://github.com/HUST-AI-HYZ/MemoryAgentBench) |
| WebShop | ②, ③ | Web | Text | [PDF](https://proceedings.neurips.cc/paper_files/paper/2022/file/82ad13ec01f9fe44c01cb91814fd7b8c-Paper-Conference.pdf) | [Code](https://github.com/princeton-nlp/WebShop) [Website](https://webshop-pnlp.github.io/) |
| Mind2Web | ②, ③ | Web | Text, Visual | [PDF](https://arxiv.org/abs/2506.21506) | [Code](https://github.com/OSU-NLP-Group/Mind2Web-2) [Website](https://osu-nlp-group.github.io/Mind2Web-2/) |
| WebArena | ②, ③ | Web | Text | [PDF](https://openreview.net/forum?id=oKn9c6ytLx) | [Code](https://github.com/web-arena-x/webarena) [Website](https://webarena.dev/) |
| VisualwebArena | ①, ②, ③ | Web | Text, Visual | [PDF](https://arxiv.org/abs/2401.13649) | [Code](https://github.com/web-arena-x/visualwebarena) [Website](https://jykoh.com/vwa) |
| AppWorld | ②, ③ | App | Text | [PDF](https://aclanthology.org/2024.acl-long.850/) | [Code](https://github.com/stonybrooknlp/appworld) [Website](https://appworld.dev/) |
| AndroidWorld | ②, ③ | GUI, App | Text, Visual | [PDF](https://openreview.net/forum?id=il5yUQsrjC) | [Code](https://github.com/google-research/android_world) |
| OSWorld | ②, ③ | GUI, OS | Text, Visual | [PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/5d413e48f84dc61244b6be550f1cd8f5-Paper-Datasets_and_Benchmarks_Track.pdf) | [Code](https://github.com/xlang-ai/OSWorld) [Website](https://os-world.github.io/) |
| Debug-Gym | ①, ③ | SWE | Text | [PDF](https://arxiv.org/abs/2503.21557) | [Code](https://github.com/microsoft/debug-gym) [Website](https://microsoft.github.io/debug-gym/) |
| MLE-Dojo | ②, ① | MLE | Text | [PDF](https://arxiv.org/abs/2505.07782) | [Code](https://github.com/MLE-Dojo/MLE-Dojo) [Website](https://mle-dojo.github.io/MLE-Dojo-page/) |
| τ-bench | ①, ③ | SWE | Text | [PDF](https://arxiv.org/abs/2506.07982) | [Code](https://github.com/sierra-research/tau2-bench) |
| TheAgentCompany | ②, ③, ⑤ | SWE | Text | [PDF](https://arxiv.org/abs/2412.14161) | [Code](https://github.com/TheAgentCompany/TheAgentCompany) [Website](https://the-agent-company.com/) |
| MedAgentGym | ① | Science | Text | [PDF](https://arxiv.org/abs/2506.04405) | [Code](https://github.com/wshi83/MedAgentGym) |
| SecRepoBench | ①, ③ | Coding, Security | Text | [PDF](https://arxiv.org/abs/2504.21205) | - |
| R2E-Gym | ①, ② | SWE | Text | [PDF](https://openreview.net/forum?id=7evvwwdo3z) | [Code](https://github.com/R2E-Gym/R2E-Gym) [Website](https://r2e-gym.github.io/) |
| HumanEval | ① | Coding | Text | [PDF](https://arxiv.org/abs/2107.03374) | [Code](https://github.com/openai/human-eval) |
| MBPP | ① | Coding | Text | [PDF](https://arxiv.org/abs/2108.07732) | [Code](https://github.com/google-research/google-research/tree/master/mbpp) |
| BigCodeBench | ① | Coding | Text | [PDF](https://openreview.net/forum?id=YrycTjllL0) | [Code](https://github.com/bigcode-project/bigcodebench) [Website](https://bigcode-bench.github.io/) |
| LiveCodeBench | ① | Coding | Text | [PDF](https://openreview.net/forum?id=chfJJYC3iL) | [Code](https://github.com/LiveCodeBench/LiveCodeBench) [Website](https://livecodebench.github.io) |
| SWE-bench | ①, ③ | SWE | Text | [PDF](https://openreview.net/forum?id=VTF8yNQM66) | [Code](https://github.com/swe-bench/SWE-bench) [Website](https://www.swebench.com/) |
| SWE-rebench | ①, ③ | SWE | Text | [PDF](https://arxiv.org/abs/2505.20411) | [Website](https://swe-rebench.com/) |
| DevBench | ②, ① | SWE | Text | [PDF](https://aclanthology.org/2025.coling-main.502/) | [Code](https://github.com/open-compass/DevEval) |
| ProjectEval | ②, ① | SWE | Text | [PDF](https://aclanthology.org/2025.findings-acl.1036/) | [Code](https://github.com/RyanLoil/ProjectEval/) [Website](https://ryanloil.github.io/ProjectEval/) |
| DA-Code | ①, ③ | Data Science, SWE | Text | [PDF](https://aclanthology.org/2024.emnlp-main.748/) | [Code](https://aclanthology.org/2024.emnlp-main.748/) [Website](https://github.com/yiyihum/da-code) |
| ColBench | ②, ①, ③ | SWE, Web Dev | Text | [PDF](https://arxiv.org/abs/2503.15478) | [Code](https://arxiv.org/abs/2503.15478) [Website](https://github.com/facebookresearch/sweet_rl) |
| NoCode-bench | ②, ① | SWE | Text | [PDF](https://arxiv.org/abs/2507.18130) | [Code](https://github.com/NoCode-bench/NoCode-bench) [Website](https://nocodebench.org/) |
| MLE-Bench | ②, ①, ③ | MLE | Text | [PDF](https://openreview.net/forum?id=6s5uXNWGIh) | [Code](https://github.com/openai/mle-bench/) [Website](https://openai.com/index/mle-bench/) |
| PaperBench | ②, ①, ③ | MLE | Text | [PDF](https://openreview.net/forum?id=xF5PuTLPbn) | [Code](https://github.com/openai/preparedness/tree/main/project/paperbench) [Website](https://openai.com/index/paperbench/) |
| Crafter | ②, ④ | Game | Visual | [PDF](https://openreview.net/forum?id=1W0z96MFEoH) | [Code](https://openreview.net/forum?id=1W0z96MFEoH) [Website](https://danijar.com/crafter) |
| Craftax | ②, ④ | Game | Visual | [PDF](https://openreview.net/forum?id=hg4wXlrQCV) | [Code](https://github.com/MichaelTMatthews/Craftax) |
| ELLM (Crafter variant) | ②, ① | Game | Visual | [PDF](https://proceedings.mlr.press/v202/du23f.html) | [Code](https://proceedings.mlr.press/v202/du23f.html) [Website](https://github.com/yuqingd/ellm) |
| SMAC / SMAC-Exp | ⑤, ② | Game | Visual | [PDF](https://arxiv.org/abs/1902.04043) | [Code](https://github.com/oxwhirl/smac) |
| Factorio | ②, ① | Game | Visual | [PDF](https://arxiv.org/abs/2503.09617) | [Code](https://github.com/JackHopkins/factorio-learning-environment) [Website](https://jackhopkins.github.io/factorio-learning-environment/) |

## Sec5.2 Frameworks

| Framework | Type | Key Features | Link | Resource |
|-----------|------|--------------|------|----------|
| **Agentic RL Frameworks** |||||
| Verifiers | Agent RL / LLM RL | Verifiable environment setup | - | [Code](https://github.com/willccbb/verifiers) |
| SkyRL-v0/v0.1 | Agent RL | Long-horizon real-world training | [Blog (v0)](https://novasky-ai.notion.site/skyrl-v0) [Blog (v0.1)](https://novasky-ai.notion.site/skyrl-v01) | [Code](https://github.com/NovaSky-AI/SkyRL) |
| AREAL | Agent RL / LLM RL | Asynchronous training | [PDF](https://openreview.net/forum?id=qJ0okaW9Z9) | [Code](https://github.com/inclusionAI/AReaL) |
| MARTI | Multi-agent RL / LLM RL | Integrated multi-agent training | - | [Code](https://github.com/TsinghuaC3I/MARTI) |
| EasyR1 | Agent RL / LLM RL | Multimodal support | - | [Code](https://github.com/hiyouga/EasyR1) |
| AgentFly | Agent RL | Scalable asynchronous execution | [PDF](https://arxiv.org/abs/2507.14897) | [Code](https://github.com/Agent-One-Lab/AgentFly) |
| Agent Lightning | Agent RL | Decoupled hierarchical RL | [PDF](https://arxiv.org/abs/2508.03680) | [Code](https://github.com/microsoft/agent-lightning) |
| **RLHF and LLM Fine-tuning Frameworks** |||||
| OpenRLHF | RLHF / LLM RL | High-performance scalable RLHF | [PDF](https://arxiv.org/abs/2405.11143) | [Code](https://github.com/OpenRLHF/OpenRLHF) |
| TRL | RLHF / LLM RL | Hugging Face RLHF | - | [Code](https://github.com/huggingface/trl) |
| trlX | RLHF / LLM RL | Distributed large-model RLHF | [PDF](https://aclanthology.org/2023.emnlp-main.530) | [Code](https://github.com/CarperAI/trlx) |
| HybridFlow | RLHF / LLM RL | Streamlined experiment management | [PDF](http://dx.doi.org/10.1145/3689031.3696075) | [Code](https://github.com/volcengine/verl) |
| SLiMe | RLHF / LLM RL | High-performance async RL | - | [Code](https://github.com/THUDM/slime) |
| **General-purpose RL Frameworks** |||||
| RLlib | General RL / Multi-agent RL | Production-grade scalable library | [PDF](https://proceedings.mlr.press/v80/liang18b.html) | [Code](https://github.com/ray-project/ray/tree/master/rllib) |
| Acme | General RL | Modular distributed components | [PDF](https://arxiv.org/abs/2006.00979) | [Code](https://github.com/google-deepmind/acme) |
| Tianshou | General RL | High-performance PyTorch platform | [PDF](https://jmlr.org/papers/v22/20-1364.html) | [Code](https://github.com/thu-ml/tianshou/) |
| Stable Baselines3 | General RL | Reliable PyTorch algorithms | [PDF](https://jmlr.org/papers/v22/20-1364.html) | [Code](https://github.com/DLR-RM/stable-baselines3) |
| PFRL | General RL | Benchmarked prototyping algorithms | [PDF](https://jmlr.org/papers/v22/20-376.html) | [Code](https://github.com/pfnet/pfrl) |

