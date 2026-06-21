# P0 素材自动匹配系统修复报告

> **日期：** 2026-06-20  
> **分支：** master  
> **触发问题：** 视频生成时 `video_source = "local"` 且无手动素材 → 标签匹配返回 0 → 视频生成失败

---

## 一、故障链路还原

```
用户生成视频 "机会成本"
  │
  ├─ [✅] LLM 生成脚本       → 成功
  ├─ [✅] TTS 生成音频       → 成功 (mimo-v2.5-tts, 51.84s)
  ├─ [✅] 字幕生成           → 成功 (edge TTS)
  │
  └─ [❌] 获取素材
        │
        ├─ material_directory = "/materials"   → 目录不存在 (Unix 路径在 Windows)
        ├─ 回退目录 storage/local_videos       → 空目录，无图片
        ├─ LLM 脚本分析                         → chars=[], emotions=[], events=[]
        │                                       → keyword="机会成本 选择 决策"
        ├─ search_materials_by_tags             → 整串匹配 "机会成本 选择 决策"
        │                                       → ❌ 0 matches
        ├─ 无 fallback 机制                     → params.video_materials 仍为 None
        └─ preprocess_video(None)               → return [] → FAILED
```

**根因定位：三个断点串联导致最终失败**

| 断点 | 文件 | 行号 | 问题 |
|---|---|---|---|
| 🔴 断点 1 | `config.toml` | 63 | `material_directory = "/materials"` 在 Windows 上不存在，回退目录也为空 |
| 🔴 断点 2 | `app/services/tagging.py` | 474-478 | keyword 整串匹配，LLM 返回的 `"机会成本 选择 决策"` 永远无法命中 |
| 🔴 断点 3 | `app/services/task.py` | 224-228 | 匹配失败后无 fallback，直接导致后续 `preprocess_video(None)` 返回空 |

---

## 二、修复内容

### 修复 1：关键字分词匹配 `tagging.py:473-487`

**问题：** `search_materials_by_tags` 中 keyword 匹配使用整串子串匹配。LLM prompt 明确要求返回 "1-3 个关键场景描述词，用空格分隔"，但搜索函数不做分词，导致 `"机会成本 选择 决策"` 必须作为完整连续文本出现在图片描述中才能命中。

**修改：**
```python
# 改前
if keyword:
    kw = keyword.strip()
    if kw and kw.lower() in tags.description.lower():
        score += 2
        match_detail["keyword_matched"] = True

# 改后
if keyword:
    keywords = [k.strip() for k in keyword.split() if k.strip()]
    matched_keywords = []
    for kw in keywords:
        if kw.lower() in tags.description.lower():
            matched_keywords.append(kw)
    if matched_keywords:
        score += len(matched_keywords)
        match_detail["keyword_matched"] = True
        match_detail["keywords_matched"] = matched_keywords
```

**效果：** `"机会成本 选择 决策"` → 拆分为 `["机会成本", "选择", "决策"]` → 每个词独立匹配 → 命中任意一个即得分。

---

### 修复 2：Random Fallback 机制 `task.py:230-256`

**问题：** 当语义匹配（characters / emotions / events / keyword）全部未命中时，`params.video_materials` 保持 `None`，`preprocess_video` 收到 `None` 直接返回空列表，导致任务失败。抽象概念脚本（如"机会成本"）天然无法从蜡笔小新素材中提取角色/情绪/事件维度，几乎必然触发此路径。

**修改：** 在 tag-based matching 返回空结果时，增加三级回退：

```
Layer 1: 语义标签匹配 (原有)
    ↓ 失败
Layer 2: 关键词拆分匹配 (修复 1)
    ↓ 失败
Layer 3: 随机采样 fallback (新增) ← 确保一定返回素材
```

实现逻辑：
1. 调用 `tagging._find_all_images(_mat_dir)` 获取目录下所有图片
2. `random.sample()` 随机选取最多 15 张
3. 构建 `MaterialInfo` 列表赋值到 `params.video_materials`
4. 记录 fallback 日志，便于排查匹配质量问题

额外改动：文件头部新增 `import random`。

---

### 修复 3：Config 启动自检 `config.py:191-208`

**问题：** `config.toml` 中配置的 `material_directory` 不存在时，原逻辑仅打印 warning，不修正配置值。下游每个调用方需要各自实现 fallback，代码分散且容易遗漏。

**修改：**
```python
# 改前
if _md:
    _normalized = _md.replace("\\", "/")
    if os.path.isdir(_normalized):
        app["material_directory"] = _normalized
    elif not os.path.isdir(_md):
        logger.warning(f"material_directory does not exist or is not accessible: {_md}")

# 改后
if _md:
    _normalized = _md.replace("\\", "/")
    if os.path.isdir(_normalized):
        app["material_directory"] = _normalized
    else:
        _fallback = os.path.join(root_dir, "storage", "local_videos")
        os.makedirs(_fallback, exist_ok=True)
        app["material_directory"] = _fallback
        logger.warning(
            f"Configured material_directory {_md!r} does not exist or is not "
            f"accessible. Automatically falling back to: {_fallback!r}. "
            f"Please update config.toml [app] material_directory to a valid path."
        )
```

**效果：** 启动时发现路径无效 → 自动创建并回退到 `storage/local_videos` → 明确提示用户修改配置。集中处理，消除下游分散的 fallback 逻辑。

---

## 三、修改文件清单

| 文件 | 改动类型 | 行号 |
|---|---|---|
| `app/services/tagging.py` | keyword 拆分匹配 | 473-487 |
| `app/services/task.py` | 新增 `import random` | 3 |
| `app/services/task.py` | random fallback 机制 | 230-256 |
| `app/config/config.py` | 启动自检 + 自动回退 | 191-208 |

---

## 四、修复后行为

```
用户生成视频 "机会成本"
  │
  ├─ material_directory = "/materials" → 不存在
  │   └─ config.py 自检 → 自动回退到 storage/local_videos ✅
  │
  ├─ LLM 脚本分析 → chars=[], emotions=[], events=[]
  │               → keyword="机会成本 选择 决策"
  │
  ├─ search_materials_by_tags
  │   └─ 拆分: ["机会成本", "选择", "决策"]
  │       └─ 逐个匹配 description → 可能命中 ✅
  │
  ├─ 如果仍未命中
  │   └─ random fallback: 随机选取 15 张图片 ✅
  │
  └─ preprocess_video(materials=[...]) → 有素材可用 ✅
```

---

## 五、遗留风险与后续建议

| 优先级 | 问题 | 建议 |
|---|---|---|
| 🟡 P1 | 抽象脚本（无角色/情绪/事件）仍然只能靠 keyword/random 匹配 | 引入 CLIP 语义向量检索（Chinese-CLIP + FAISS） |
| 🟡 P1 | 素材库为空时 random fallback 也会失败 | 启动时检查素材目录是否有图片，提前警告 |
| 🟢 P2 | 随机 fallback 素材质量不可控 | 增加"万能素材"分类，优先从该分类 fallback |
| 🟢 P2 | 标签同义词未展开（如"开心""快乐""高兴"） | 增加同义词映射，匹配时展开 |
