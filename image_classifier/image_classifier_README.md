# ImageClassifier - 通用图片分类工具

将图片收集、关键词分类、特征分类和LLM分类整合为一个可复用的类。

## 快速开始

```python
from image_classifier import ImageClassifier

# 1. 创建实例
clf = ImageClassifier(
    source_dir=r"D:\项目\图片目录",
    categories={
        "实验结果": ["mea", "result", "exp"],
        "理论图": ["sim", "theory"],
        "流程图": ["flow", "process"],
    },
    project_context="这是一个关于超表面设计的研究项目...",
)

# 2. 收集图片并按日期重命名
clf.organize_with_dates(target_dir=r"D:\项目\图片收集")

# 3. 按关键词分类（文件名匹配）
results = clf.run_keyword_classification()

# 4. 按图像特征分类（OpenCV分析）
results = clf.run_feature_classification()

# 5. 使用LLM智能分类（需要Ollama）
results = clf.run_llm_classification(move_files=True)

# 6. 生成报告
clf.generate_report(results)
```

## 类初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `source_dir` | str | 必填 | 图片根目录 |
| `categories` | dict | 6组默认分类 | `{分类名: [关键词列表]}` |
| `llm_model` | str | `"qwen3.5:9b"` | Ollama模型名称 |
| `llm_url` | str | `http://localhost:11434/api/generate` | Ollama API地址 |
| `llm_timeout` | int | 600 | LLM请求超时秒数 |
| `project_context` | str | `""` | 项目背景描述（LLM分类用） |

## 方法一览

### 收集与整理

| 方法 | 说明 |
|------|------|
| `collect_images(root_dir?)` | 递归收集目录下所有图片路径 |
| `organize_with_dates(target_dir?, copy=True)` | 收集图片并添加修改日期后缀，`copy=True`复制/`False`移动 |

### 分类

| 方法 | 说明 |
|------|------|
| `classify_by_keywords(filename)` | 根据文件名关键词返回分类 |
| `run_keyword_classification(target_dir?)` | 关键词分类并移动文件 |
| `analyze_image_features(img_path)` | OpenCV分析图像特征，返回dict |
| `classify_by_features(features, filename)` | 根据特征+文件名返回分类 |
| `run_feature_classification(unclass_dir?)` | 特征分类未分类图片 |
| `run_llm_classification(unclass_dir?, move_files=True, sleep_interval=1.0)` | LLM分类，可自动移动文件 |

### 工具

| 方法 | 说明 |
|------|------|
| `test_ollama_connection()` | 测试Ollama API连接 |
| `generate_report(results, output_path?)` | 生成Markdown分类报告 |

## 分类优先级

```
关键词匹配 > 图像特征分析 > 默认分类（未分类）
```

特征分类规则：
- `edge > 0.1 且 hues < 5` → 流程图
- `edge > 0.1 且 hues >= 5` → 结构示意图
- `edge < 0.01 且 hues > 20` → AI生成图
- `brightness < 50` → 实验结果图片

## 自定义分类示例

```python
# 科研论文分类
clf = ImageClassifier(
    source_dir=r"/path/to/images",
    categories={
        "实验数据": ["experiment", "measurement", "data"],
        "模拟结果": ["simulation", "fem", "cfd"],
        "设计图纸": ["cad", "drawing", "blueprint"],
        "文档截图": ["screenshot", "capture"],
    },
    default_category="待审核",
)

# 网络图片整理
clf = ImageClassifier(
    source_dir=r"/path/to/downloads",
    categories={
        "风景": ["landscape", "nature", "sunset"],
        "人物": ["portrait", "face", "selfie"],
        "动物": ["cat", "dog", "bird", "animal"],
        "建筑": ["building", "architecture", "city"],
    },
    default_category="其他",
)

# 仅使用LLM分类（不依赖关键词）
clf = ImageClassifier(
    source_dir=r"/path/to/images",
    categories={"A类": [], "B类": [], "C类": []},
    project_context="请根据图片内容分为A/B/C三类...",
)
clf.test_ollama_connection()  # 先测试连接
results = clf.run_llm_classification()
```

## 依赖

```bash
pip install opencv-python numpy requests
```

Ollama LLM分类额外需要：
- [Ollama](https://ollama.com) 已安装并运行
- 已拉取对应模型：`ollama pull qwen3.5:9b`