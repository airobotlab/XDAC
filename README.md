# XDAC | [Homepage](https://airobotlab.github.io/XDAC/) | [Github](https://github.com/airobotlab/XDAC) | [Dataset & Code](https://huggingface.co/keepsteady/XDAC_obs) | [Colab-tutorial](https://colab.research.google.com/drive/1qMCv5SDEc7zshg4m2xyT91FQFDlJtvb8?usp=sharing) |

Official code, models, and dataset for the paper ["XDAC is an XAI-driven framework for detecting and attributing LLM-generated comments in Korean news"](https://github.com/airobotlab/XDAC/blob/main/paper/250611_XDAC_ACL2025_camera_ready.pdf), [Association for Computational Linguistics (ACL 2025) main conference](https://2025.aclweb.org/).

## Abstract
Large language models (LLMs) generate human-like text, raising concerns about their misuse in creating deceptive content. Detecting LLM-generated comments (LGC) in online news is essential for preserving online discourse integrity and preventing opinion manipulation. However, effective detection faces two key challenges; the brevity and informality of news comments limit traditional methods, and the absence of a publicly available LGC dataset hinders model training, especially for languages other than English. To address these challenges, we propose a twofold approach. First, we develop an LGC generation framework to construct a high-quality dataset with diverse and complex examples. Second, we introduce XDAC (**X**AI-Driven **D**etection and **A**ttribution of LLM-Generated **C**omments), a framework utilizing explainable AI, designed for the detection and attribution of short-form LGC in Korean news articles. XDAC leverages XAI to uncover distinguishing linguistic patterns at both token and character levels. We present the first large-scale benchmark dataset, comprising 1.3M human-written comments from Korean news platforms and 1M LLM-generated comments from 14 distinct models. XDAC outperforms existing methods, achieving a 98.5\% F1 score in LGC detection with a relative improvement of 68.1\%, and an 84.3\% F1 score in attribution.


## Download Links for Dataset & Models
- [Dataset & Code (Hugging Face)](https://huggingface.co/keepsteady/XDAC_obs)
- The dataset consists of LLM-generated comments, and all user information is anonymized by replacing identifiable details with “XXX” to protect individual privacy.
- Note: Access to the dataset and models is protected via Hugging Face's gated access system.
 - To download the resources, users must:
  - Log in to their Hugging Face account
  - Request access by providing their email and username



### Colab Tutorial
 - [Inference-Code-Link](https://colab.research.google.com/drive/1qMCv5SDEc7zshg4m2xyT91FQFDlJtvb8?usp=sharing)

### Install Dependencies
```bash
pip install torch transformers captum
```

### Python code with Pipeline
```python
## 0) download XDAC model/data from huggingface (90s)
import os
XDAC_root_path = './XDAC_obs'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="keepsteady/XDAC_obs",
    local_dir=XDAC_root_path,
    local_dir_use_symlinks=False
)
from XDAC_obs.xdac_encrypted import AIUnifiedEngine, get_xdac_path
print("✅ Secure XDAC imported successfully!")
XDAC_root_path = get_xdac_path()


## 1) Load Korean LGC dataset
import json
from datasets import Dataset
from pprint import pprint

path_data = os.path.join(XDAC_root_path, './LGC_data/LGC_data_v1.0.json')

with open(path_data, 'r', encoding='utf-8') as f:
    data_list = json.load(f)
    dataset_LGC = Dataset.from_list(data_list)

print(dataset_LGC)
pprint(dataset_LGC[-1])


## 2) Load XDAC
print("XDAC Unified Engine: AI Detection & Attribution")
print("=" * 60)

device = 'cuda'

unified_engine = AIUnifiedEngine(
    detection_model_path=os.path.join(XDAC_root_path, 'XDAC-D'),
    attribution_model_path=os.path.join(XDAC_root_path, 'XDAC-A'),
    device=device,
    xai_enabled=True
)



## 3) Run XDAC
multiple_texts = [
  '서울대도 옮기고 싶냐? 대체 어디까지 욕심 부릴 거냐?',
  '17조 수출에 12조 지원이라니! 🤔 거의 뭐 퍼주는 수준 아닌가?! 그래도 국뽕 차오르네 🤣',
  '세종 투기꾼들 또 설레발 치는 거 아냐? 진짜 짜증난다.',
  '이런 공약으로 표 얻으려는 게 너무 뻔히 보인다.',
  '김 전 XX관의 사퇴는 옳은 결정이었어요. 자녀의 학교폭력 문제에 대한 진상이 조만간 밝혀지길 바라요.',
  '똑바로 좀 해라 똑바로 어??   잘 해봐 좀!!!',
  '염병 ㅋㅋㅋㅋzzzzzzz   놀고 앉았네',
  '이게 참말로 말이됀다구? 이거 조작 아냐???',
  '빠람빠빠밤 빠라빠라빰 빠라라라라~~~',
  '와나 \n\n\n진짜 어이털리네???? 이거 조작 아냐??',
]

# Unified inference
results = unified_engine.predict(multiple_texts, batch_size=10)
unified_engine.print_results(results, save_path='result_XDAC.txt')
unified_engine.save_xai_results_to_html(results, html_file_path='result_XDAC.html')

# Get top predictions for detailed analysis
top_results = unified_engine.get_top_predictions(results, top_k=3)
```

## Citation
- This part will be updated once the official BibTeX is released.
```bibtex
@proceedings{acl-2025-long,
    title = "Proceedings of the 63nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    editor = "Wooyoung Go and
      Hyoungshick Kim and
      Alice Oh and
      Yongdae Kim",
    month = July,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.0/"
}
```

## Contact
 - Wooyoung Go, `gwy876@gmail.com`
