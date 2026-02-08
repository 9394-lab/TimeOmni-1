<div align="center">
<img src="figs/logo.png" alt="Logo" width="120"/>

<h1><b>
(ICLR'26) TimeOmni-1: Incentivizing Complex Reasoning with Time Series in Large Language Models
</b></h1>


<p align="left">
  <a href="https://arxiv.org/abs/2509.24803">
    <img
      src="https://img.shields.io/badge/TimeOmni--1-Paper-red?logo=arxiv&logoColor=red"
      style="display: inline-block; vertical-align: middle;"
      alt="TimeOmni-1 Paper on arXiv"
    />
  </a>

  <a href="https://huggingface.co/anton-hugging/TimeOmni-1-7B">
    <img
      src="https://img.shields.io/badge/TimeOmni--1-Model-yellow?logo=huggingface&logoColor=white"
      style="display: inline-block; vertical-align: middle;"
      alt="TimeOmni-1 Model on Hugging Face"
    />
  </a>

  <a href="https://huggingface.co/datasets/anton-hugging/timeomni-1-testbed">
    <img
      src="https://img.shields.io/badge/TimeOmni--1-Dataset-orange?logo=huggingface&logoColor=white"
      style="display: inline-block; vertical-align: middle;"
      alt="TimeOmni-1 Dataset on Hugging Face"
    />
  </a>

  <a href="https://huggingface.co/spaces/anton-hugging/TimeOmni-1">
    <img
      src="https://img.shields.io/badge/TimeOmni--1-Demo-blue?logo=huggingface&logoColor=white"
      style="display: inline-block; vertical-align: middle;"
      alt="TimeOmni-1 Demo on Hugging Face Spaces"
    />
  </a>

  <a href="https://github.com/9394-lab/TimeOmni-1" target="_blank" style="margin: 2px;">
    <img
      src="https://img.shields.io/badge/TimeOmni--1-Inference%20Code-536af5?logo=github&logoColor=white"
      style="display: inline-block; vertical-align: middle;"
      alt="TimeOmni-1 Inference Code on GitHub"
    />
  </a>
</p>

</div>

**This repository provides installation and usage scripts for TimeOmni-1.**

---

## 🛠️ Installation
```bash
conda create -n timeomni python=3.10
conda activate timeomni
pip install -r requirements.txt
```

## 📦 Model Download
```bash
python install/download_hf_model.py
```
Default model path: `~/.cache/huggingface/hub`.

## 🧪 Dataset Download
```bash
python install/download_testbed.py
```
This creates:
- `data/timeomni1_id_test.json`
- `data/timeomni1_ood_test.json`

## 🚀 Inference (single question)
Default system prompt:
```
Output Format:
<think>Your step-by-step reasoning process that justifies your answer</think>
<answer>Your final answer(Note: Only output a single uppercase letter of the correct option)</answer>
```

Run:
```bash
python inference/inference.py \
  --model_dir "Local Model Path /models--anton-hugging--TimeOmni-1-7B/snapshots/<hash>" \
  --question "Your Question" \
  --system_prompt "Output Format:\n<think>Your step-by-step reasoning process that justifies your answer</think>\n<answer>Your final answer(Note: Only output a single uppercase letter of the correct option)</answer>"
```

## 📊 Evaluation
```bash
bash eval/run-timeomini_test.sh
```
Optional env overrides:
```bash
MODEL_DIR=anton-hugging/TimeOmni-1-7B \
ANS_ID_PATH=answer/timeomni1_test/your_id_outputs.json \
RES_ID_PATH=answer/timeomni1_test/your_id_results.json \
ANS_OOD_PATH=answer/timeomni1_test/your_ood_outputs.json \
RES_OOD_PATH=answer/timeomni1_test/your_ood_results.json \
bash eval/run-timeomini_test.sh
```

We report Success Rate (SR), defined as the proportion of model outputs that yield a valid and extractable answer. All other metrics are computed on valid cases only.

**Tasks 1, 2, 4:** model outputs a single uppercase letter (A/B/C/D). Metric: Accuracy (ACC).  
**Task 3:** model outputs a sequence (e.g., `[2, 20, 21, ..., 83]`). Metric: Mean Absolute Error (MAE).

## ✍️ Citation

```bibtex
@article{guan2025timeomni,
  title={Timeomni-1: Incentivizing complex reasoning with time series in large language models},
  author={Guan, Tong and Meng, Zijie and Li, Dianqi and Wang, Shiyu and Yang, Chao-Han Huck and Wen, Qingsong and Liu, Zuozhu and Siniscalchi, Sabato Marco and Jin, Ming and Pan, Shirui},
  journal={arXiv preprint arXiv:2509.24803},
  year={2025}
}
```