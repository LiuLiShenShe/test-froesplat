from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import requests

from direct_common import SWITCHES, WORK, relative, sha256, write_json


def main():
    secret = os.environ.get("AGNES_API_KEY", "")
    if not secret:
        raise ValueError("SetAGNES_API_KEY without committingcredentials")
    paths = [WORK / "figures" / f"comparison_{stem}.png" for stem in SWITCHES + ["WanNianQing1_0223"]]
    paths.append(WORK / "figures/paired_and_raster_sensitivity.png")
    result = []
    for path in paths:
        prompt = ("只做科学图件排版和文字质量检查，不做人工GT认证。客观描述面板、标题、图例、图片中显示的文字和数值。"
                  "检查文字截断、遮挡、面板缺失、黑白掩膜可见性和几何歧义提示。不要修改图片，不判断人工标注是否准确，不推测真人签名。"
                  "所有图像来自真实实验，标题已给出原始分辨率非空掩膜面积；不得因F1=0说空掩膜。"
                  "若白色区域很小看不清，请说显示尺度有限，不猜测全黑。只做文字排版识别，实验真值以CSV为准。回答中文。")
        payload = {"model": "agnes-2.5-flash", "messages": [{"role": "user", "content": [
                   {"type": "text", "text": prompt}, {"type": "image_url", "image_url": {
                   "url": "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()}}]}]}
        try:
            response = requests.post("https://apihub.agnes-ai.cn/v1/chat/completions",
                       headers={"Authorization": "Bearer " + secret, "Content-Type": "application/json"}, json=payload, timeout=180)
            if response.status_code != 200:
                result.append({"path": relative(path), "image_sha256": sha256(path), "status": "FAILED", "http_status": response.status_code})
                continue
            answer = response.json()["choices"][0]["message"]["content"]
            result.append({"path": relative(path), "image_sha256": sha256(path), "status": "OK", "description": answer,
                           "source": "图片内容由Agnes API识别; machinevisualQA only, nothumanreview"})
        except (requests.RequestException, ValueError, KeyError) as error:
            result.append({"path": relative(path), "image_sha256": sha256(path), "status": "FAILED", "error_type": type(error).__name__})
    write_json(WORK / "logs/AGNES_VISUAL_QA.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
