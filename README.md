# lab-tools

Universal image classifier with keyword matching, OpenCV feature analysis, and LLM-powered classification.

## Features

- **Keyword Classification** - Match files by filename keywords
- **Feature Classification** - Analyze image features with OpenCV (edges, hues, brightness)
- **LLM Classification** - Use Ollama-powered LLM for intelligent classification
- **Report Generation** - Generate markdown classification reports

## Quick Start

```python
from image_classifier import ImageClassifier

clf = ImageClassifier(
    source_dir=r"D:\项目\图片目录",
    categories={
        "实验结果": ["mea", "result", "exp"],
        "理论图": ["sim", "theory"],
        "流程图": ["flow", "process"],
    },
    project_context="这是一个关于超表面设计的研究项目...",
)

# Keyword classification
results = clf.run_keyword_classification()

# Feature classification
results = clf.run_feature_classification()

# LLM classification (requires Ollama)
results = clf.run_llm_classification(move_files=True)

# Generate report
clf.generate_report(results)
```

## Dependencies

```bash
pip install opencv-python numpy requests
```

For LLM classification, install [Ollama](https://ollama.com) and pull a model:
```bash
ollama pull qwen3.5:9b
```

## License

MIT
