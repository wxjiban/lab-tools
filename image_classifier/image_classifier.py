import os
import re
import json
import base64
import shutil
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Optional

import cv2
import numpy as np
import requests


class ImageClassifier:
    """通用图片分类工具，支持关键词分类、特征分类和LLM分类"""

    IMAGE_EXTENSIONS = {
        ".jpg", ".jpeg", ".png", ".gif", ".bmp",
        ".webp", ".svg", ".tiff", ".tif",
    }

    def __init__(
        self,
        source_dir: str,
        categories: Optional[dict] = None,
        llm_model: str = "qwen3.5:9b",
        llm_url: str = "http://localhost:11434/api/generate",
        llm_timeout: int = 600,
        project_context: str = "",
    ):
        self.source_dir = Path(source_dir)
        self.llm_model = llm_model
        self.llm_url = llm_url
        self.llm_timeout = llm_timeout
        self.project_context = project_context

        self.categories = categories or {
            "论文流程图": [
                "compare", "comparison", "process", "flow",
                "pipeline", "workflow",
            ],
            "论文素材库图片": [
                "figure", "fig", "image", "pic",
                "diagram", "chart", "graph", "plot",
            ],
            "实验结果图片": [
                "mea", "measured", "exp", "result",
                "heatmap", "purity", "mag", "phase",
                "farfield", "fieldcut",
            ],
            "理论结果图": [
                "sim", "simulated", "theory",
                "theoretical", "calc", "predicted",
            ],
            "结构示意图": [
                "structure", "layout", "cell", "unit",
                "metaatom", "array", "distribution",
                "sample", "design", "strategy", "encoding",
                "sequence", "dartboard", "dual", "single",
                "hex", "slant", "ybeams", "multibeams",
            ],
            "理论示意图": [
                "schematic", "concept", "illustration",
                "sketch", "model", "simulation",
            ],
        }
        self.default_category = "未分类图片"

    # ==================== 收集与整理 ====================

    def collect_images(self, root_dir: Optional[str] = None) -> list:
        root = Path(root_dir) if root_dir else self.source_dir
        images = []
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                if Path(f).suffix.lower() in self.IMAGE_EXTENSIONS:
                    images.append(str(Path(dirpath) / f))
        return images

    def organize_with_dates(
        self,
        target_dir: Optional[str] = None,
        copy: bool = True,
    ) -> dict:
        target = Path(target_dir) if target_dir else self.source_dir / "collected_images"
        target.mkdir(parents=True, exist_ok=True)

        all_images = self.collect_images(self.source_dir)
        results = {"total": 0, "success": 0, "errors": []}
        op = shutil.copy2 if copy else shutil.move

        for src in all_images:
            if str(src).startswith(str(target)):
                continue
            results["total"] += 1
            name, ext = os.path.splitext(os.path.basename(src))
            base = self._strip_date_suffix(name)
            mtime = os.path.getmtime(src)
            date_sfx = datetime.fromtimestamp(mtime).strftime("_%Y%m%d")
            new_name = base + date_sfx + ext
            dst = target / new_name
            dst = self._resolve_conflict(dst)
            try:
                op(src, dst)
                results["success"] += 1
            except Exception as e:
                results["errors"].append((src, str(e)))

        return results

    # ==================== 关键词分类 ====================

    def classify_by_keywords(self, filename: str) -> str:
        name_lower = filename.lower()
        for category, keywords in self.categories.items():
            for kw in keywords:
                if kw in name_lower:
                    return category
        return self.default_category

    def run_keyword_classification(self, target_dir: Optional[str] = None) -> dict:
        src = Path(target_dir) if target_dir else self.source_dir
        all_categories = list(self.categories.keys()) + [self.default_category]
        for cat in all_categories:
            (src / cat).mkdir(exist_ok=True)

        files = [f for f in os.listdir(src) if (src / f).is_file()]
        results = defaultdict(list)

        for filename in files:
            cat = self.classify_by_keywords(filename)
            results[cat].append(filename)
            cat_dir = src / cat
            src_path = src / filename
            dst_path = cat_dir / filename
            try:
                shutil.move(str(src_path), str(dst_path))
            except Exception as e:
                print(f"移动失败 {filename}: {e}")

        return dict(results)

    # ==================== 特征分类 ====================

    def analyze_image_features(self, img_path: str) -> Optional[dict]:
        try:
            img = cv2.imread(img_path)
            if img is None:
                return None
            h, w, c = img.shape
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (h * w)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            unique_hues = len(np.unique(hsv[:, :, 0] // 10))
            brightness = float(np.mean(gray))
            return {
                "width": w, "height": h, "channels": c,
                "edge_density": edge_density,
                "unique_hues": unique_hues,
                "brightness": brightness,
            }
        except Exception as e:
            print(f"分析错误 {img_path}: {e}")
            return None

    def classify_by_features(self, features: Optional[dict], filename: str) -> str:
        cat = self.classify_by_keywords(filename)
        if cat != self.default_category:
            return cat

        if features is None:
            return self.default_category

        edge = features["edge_density"]
        hues = features["unique_hues"]
        bright = features["brightness"]

        if edge > 0.1:
            return "论文流程图" if hues < 5 else "结构示意图"
        if edge < 0.01 and hues > 20:
            return "AI生成图"
        if bright < 50:
            return "实验结果图片"

        return self.default_category

    def run_feature_classification(
        self,
        unclass_dir: Optional[str] = None,
    ) -> list:
        src = Path(unclass_dir) if unclass_dir else self.source_dir / self.default_category
        if not src.exists():
            print(f"目录不存在: {src}")
            return []

        files = [
            f for f in os.listdir(src)
            if (src / f).is_file() and Path(f).suffix.lower() in self.IMAGE_EXTENSIONS
        ]
        results = []
        for filename in files:
            filepath = str(src / filename)
            features = self.analyze_image_features(filepath)
            cat = self.classify_by_features(features, filename)
            results.append({"filename": filename, "category": cat, "features": features})
            print(f"{filename} -> {cat}")

        self._print_stats(results)
        return results

    # ==================== LLM 分类 ====================

    def test_ollama_connection(self) -> bool:
        try:
            resp = requests.post(
                self.llm_url,
                json={"model": self.llm_model, "prompt": "Hello", "stream": False},
                timeout=10,
            )
            if resp.status_code == 200:
                print(f"连接成功: {resp.json().get('response', '')[:100]}")
                return True
            print(f"HTTP {resp.status_code}: {resp.text}")
            return False
        except Exception as e:
            print(f"连接失败: {e}")
            return False

    def _image_to_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _call_ollama(self, image_base64: str) -> str:
        category_list = "\n".join(
            f"{i+1}. {cat}" for i, cat in enumerate(self.categories.keys())
        )
        prompt = f"""你是一个图像分类和描述专家。根据以下项目背景和图片内容，对图片进行分类并描述。

项目背景：
{self.project_context}

请将图片分类到以下类别之一（只输出类别名称）：
{category_list}

然后描述图片内容（包括图片中的关键元素、数据、图表、文字等）。

请按以下格式输出：
类别：[类别名称]
描述：[图片内容描述]"""

        data = {
            "model": self.llm_model,
            "prompt": prompt,
            "stream": False,
            "images": [image_base64],
        }
        try:
            resp = requests.post(self.llm_url, json=data, timeout=self.llm_timeout)
            if resp.status_code == 200:
                return resp.json().get("response", "")
            return f"错误: HTTP {resp.status_code}"
        except requests.exceptions.Timeout:
            return "错误: 请求超时"
        except Exception as e:
            return f"错误: {str(e)}"

    def run_llm_classification(
        self,
        unclass_dir: Optional[str] = None,
        move_files: bool = True,
        sleep_interval: float = 1.0,
    ) -> list:
        src = Path(unclass_dir) if unclass_dir else self.source_dir / self.default_category
        if not src.exists():
            print(f"目录不存在: {src}")
            return []

        files = [
            f for f in os.listdir(src)
            if (src / f).is_file() and Path(f).suffix.lower() in self.IMAGE_EXTENSIONS
        ]
        print(f"找到 {len(files)} 个未分类图片")

        results = []
        for i, filename in enumerate(files):
            filepath = str(src / filename)
            print(f"处理 ({i+1}/{len(files)}): {filename}")
            try:
                b64 = self._image_to_base64(filepath)
                response = self._call_ollama(b64)
                category, description = self._parse_llm_response(response)
                results.append({
                    "filename": filename,
                    "category": category,
                    "description": description,
                    "full_response": response,
                })
                print(f"  类别: {category}")
            except Exception as e:
                results.append({
                    "filename": filename,
                    "category": "错误",
                    "description": str(e),
                    "full_response": "",
                })
            time.sleep(sleep_interval)

        self._save_results_txt(results, src.parent / "图片分类描述.txt")
        self._update_report_md(results, src.parent / "图片分类报告.md")

        if move_files:
            self._move_classified_files(results, src)

        return results

    # ==================== 报告生成 ====================

    def generate_report(
        self,
        results: list,
        output_path: Optional[str] = None,
    ) -> str:
        out = Path(output_path) if output_path else self.source_dir / "图片分类报告.md"
        lines = [
            "# 图片分类报告\n",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"源目录: `{self.source_dir}`\n",
            f"总文件数: {len(results)}\n",
            "\n## 分类统计\n",
            "| 分类 | 文件数 | 示例文件 |",
            "|------|--------|----------|",
        ]
        stats = defaultdict(list)
        for r in results:
            stats[r["category"]].append(r["filename"])
        for cat, files in stats.items():
            examples = ", ".join(files[:3])
            lines.append(f"| {cat} | {len(files)} | {examples} |")
        lines.append("\n## 各分类详情\n")
        for cat, files in stats.items():
            lines.append(f"### {cat} ({len(files)} 个文件)\n")
            for f in files:
                lines.append(f"- {f}")
            lines.append("")

        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"报告已生成: {out}")
        return str(out)

    # ==================== 内部方法 ====================

    @staticmethod
    def _strip_date_suffix(name: str) -> str:
        return re.sub(r"[_\s]?(\d{4}[-_]?\d{2}[-_]?\d{2})$", "", name, flags=re.IGNORECASE)

    @staticmethod
    def _resolve_conflict(path: Path) -> Path:
        if not path.exists():
            return path
        stem, ext = path.stem, path.suffix
        parent = path.parent
        counter = 1
        while True:
            new_path = parent / f"{stem}_{counter}{ext}"
            if not new_path.exists():
                return new_path
            counter += 1

    def _parse_llm_response(self, text: str) -> tuple:
        category = self.default_category
        description = "无描述"
        for line in text.strip().split("\n"):
            line = line.strip()
            if line.startswith("类别："):
                raw_cat = line.replace("类别：", "").strip()
                for known in self.categories:
                    if known in raw_cat:
                        category = known
                        break
                else:
                    category = raw_cat
            elif line.startswith("描述："):
                description = line.replace("描述：", "").strip()
        return category, description

    def _move_classified_files(self, results: list, src_dir: Path):
        moved = 0
        for r in results:
            cat = r["category"]
            if cat not in self.categories and cat != self.default_category:
                target_name = None
                for key in self.categories:
                    if key in cat:
                        target_name = key
                        break
            else:
                target_name = cat
            if not target_name:
                continue
            target_dir = src_dir.parent / target_name
            target_dir.mkdir(exist_ok=True)
            src_path = src_dir / r["filename"]
            dst_path = target_dir / r["filename"]
            try:
                shutil.move(str(src_path), str(dst_path))
                print(f"移动: {r['filename']} -> {target_name}")
                moved += 1
            except Exception as e:
                print(f"移动失败 {r['filename']}: {e}")
        print(f"共移动了 {moved} 个文件到对应分类文件夹")

    @staticmethod
    def _save_results_txt(results: list, output_path: Path):
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("图片分类和描述结果\n" + "=" * 50 + "\n\n")
            for r in results:
                f.write(f"文件: {r['filename']}\n")
                f.write(f"类别: {r['category']}\n")
                f.write(f"描述: {r['description']}\n")
                f.write(f"完整响应:\n{r['full_response']}\n" + "-" * 50 + "\n")

    def _update_report_md(self, results: list, report_path: Path):
        mode = "a" if report_path.exists() else "w"
        with open(report_path, mode, encoding="utf-8") as f:
            f.write(f"\n\n## AI分类结果（基于ollama {self.llm_model}模型）\n\n")
            for r in results:
                f.write(f"### {r['filename']}\n")
                f.write(f"- **AI分类**: {r['category']}\n")
                f.write(f"- **AI描述**: {r['description']}\n\n")

    @staticmethod
    def _print_stats(results: list):
        print("\n=== 分类统计 ===")
        stats = defaultdict(list)
        for r in results:
            stats[r["category"]].append(r["filename"])
        for cat, files in stats.items():
            print(f"{cat}: {len(files)} 个文件")
            for f in files[:3]:
                print(f"  {f}")
            if len(files) > 3:
                print(f"  ... 等 {len(files)} 个文件")