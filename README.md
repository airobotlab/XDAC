# XDAC | [Homepage](https://www.bllossom.ai/) | [Github](https://github.com/airobotlab/XDAC) | [Colab-tutorial](https://colab.research.google.com/drive/1fBOzUVZ6NRKk_ugeoTbAOokWKqSN47IG?usp=sharing) |

Official code, models, and dataset for the paper ["XDAC is an XAI-driven framework for detecting and attributing LLM-generated comments in Korean news"](https://arxiv.org/abs/2411.04822), Association for Computational Linguistics (ACL 2025).

## Abstract
Large language models (LLMs) generate human-like text, raising concerns about their misuse in creating deceptive content. Detecting LLM-generated comments (LGC) in online news is essential for preserving online discourse integrity and preventing opinion manipulation. However, effective detection faces two key challenges; the brevity and informality of news comments limit traditional methods, and the absence of a publicly available LGC dataset hinders model training, especially for languages other than English. To address these challenges, we propose a twofold approach. First, we develop an LGC generation framework to construct a high-quality dataset with diverse and complex examples. Second, we introduce XDAC (**X**AI-Driven **D**etection and **A**ttribution of LLM-Generated **C**omments), a framework utilizing explainable AI, designed for the detection and attribution of short-form LGC in Korean news articles. XDAC leverages XAI to uncover distinguishing linguistic patterns at both token and character levels. We present the first large-scale benchmark dataset, comprising 1.3M human-written comments from Korean news platforms and 1M LLM-generated comments from 14 distinct models. XDAC outperforms existing methods, achieving a 98.5\% F1 score in LGC detection with a relative improvement of 68.1\%, and an 84.3\% F1 score in attribution. To validate real-world applicability, we analyze 5.24M news comments from Naver, South Korea's leading online news platform, identifying 27,029 potential LLM-generated comments.

## Example code

### Colab Tutorial
 - [Inference-Code-Link](https://colab.research.google.com/drive/1fBOzUVZ6NRKk_ugeoTbAOokWKqSN47IG?usp=sharing)

### Install Dependencies
```bash
pip install torch transformers==4.40.0 accelerate
```

### Python code with Pipeline
```python
import transformers
import torch

model_id = "MLP-KTLim/llama3-Bllossom"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

pipeline.model.eval()

PROMPT = '''당신은 유용한 AI 어시스턴트입니다. 사용자의 질의에 대해 친절하고 정확하게 답변해야 합니다.
You are a helpful AI assistant, you'll need to answer users' queries in a friendly and accurate manner.'''
instruction = "서울과학기술대학교 MLP연구실에 대해 소개해줘"

messages = [
    {"role": "system", "content": f"{PROMPT}"},
    {"role": "user", "content": f"{instruction}"}
    ]

prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
    prompt,
    max_new_tokens=2048,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
    repetition_penalty = 1.1
)

print(outputs[0]["generated_text"][len(prompt):])

# 서울과학기술대학교 MLP연구실은 멀티모달 자연어처리 연구를 하고 있습니다. 구성원은 임경태 교수와 김민준, 김상민, 최창수, 원인호, 유한결, 임현석, 송승우, 육정훈, 신동재 학생이 있습니다.
```

## Citation

```bibtex
@proceedings{acl-2024-long,
    title = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.0/"
}
```

## Contact
 - Wooyoung Go, `gwy876@gmail.com`
