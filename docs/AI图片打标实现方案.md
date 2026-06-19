# AI 图片打标 — 完整实现方案

## 1. 需求背景

本地素材库目前仅按文件名和目录结构管理图片（`.jpg` / `.jpeg` / `.png` / `.bmp`）。本项目的核心使用场景是**蜡笔小新动画截图素材**的管理与检索。当前靠文件名记忆的方式效率低下，无法按「角色」「情绪」「事件」等维度快速定位所需素材。因此需要引入 AI 图片打标能力，自动为每张截图生成结构化的语义标签。

### 核心目标

1. **自动打标**：对素材库中的每张蜡笔小新截图，调用 Qwen3-VL-Flash 视觉模型，自动识别角色、情绪、事件、画面描述、主色调
2. **增量更新**：仅对新增或变更的图片进行打标，已打标图片通过 MD5 哈希校验跳过
3. **标签检索**：在 WebUI 中支持按角色、情绪、事件、描述关键词搜索/筛选素材
4. **脚本匹配**：生成视频时，基于标签自动匹配与脚本内容最相关的图片素材

---

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit WebUI                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ 素材管理页面  │  │ 打标管理页面  │  │ 标签搜索组件  │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
└─────────┼─────────────────┼─────────────────┼───────────┘
          │                 │                 │
          ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────┐
│                    FastAPI 后端                           │
│  ┌──────────────────────────────────────────────────┐   │
│  │  /api/v1/materials/tags  (标签相关接口)           │   │
│  │    GET    /tags             — 标签统计概览        │   │
│  │    POST   /tags/generate    — 触发批量打标任务    │   │
│  │    GET    /tags/status      — 查询打标进度        │   │
│  │    GET    /tags/search      — 按标签搜索素材      │   │
│  │    DELETE /tags             — 清除指定素材的标签  │   │
│  └──────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────┐   │
│  │  app/services/tagging.py  (打标核心服务)          │   │
│  │    - 图片哈希计算 (MD5)                            │   │
│  │    - Qwen3-VL-Flash Vision API 调用               │   │
│  │    - 标签 sidecar JSON 读写                        │   │
│  │    - 批量任务调度 + 进度上报                        │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│                    数据存储层                             │
│ 素材库目录:  /materials/  (或 storage/local_videos/)      │
│   ├── 001.jpg                    ← 原始图片              │
│   ├── 001.jpg.tags.json          ← 标签 sidecar 文件     │
│   ├── 002.jpg                                             │
│   ├── 002.jpg.tags.json                                   │
│   └── ...                                                 │
└─────────────────────────────────────────────────────────┘
```

### 设计决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 标签存储 | Sidecar JSON 文件 | 无需引入数据库；与图片同目录，便于迁移；人类可读可编辑 |
| 标签范式 | 封闭枚举（characters / emotions / events） | 保证标签一致性，AI 只做选择题不做作文题 |
| Vision 模型 | Qwen3-VL-Flash | 阿里云 DashScope 托管，OpenAI 兼容接口，成本低、速度快 |
| 调用方式 | OpenAI 兼容接口 + base64 图片 | 复用项目已有的 `openai` SDK，无需新依赖 |
| 哈希校验 | MD5 文件内容摘要 | 快速判断图片是否变更，避免重复打标 |
| 批量处理 | ThreadPoolExecutor（有界线程池） | 控制并发，避免触发 API rate limit |
| 进度追踪 | 复用现有 MemoryState / RedisState | 与任务系统保持一致 |

---

## 3. 标签数据模型

### 3.1 Pydantic Schema（`app/models/schema.py` 新增）

```python
class ImageTags(BaseModel):
    """单张蜡笔小新截图的 AI 标签"""

    # ========== 基础信息 ==========
    file_path: str                    # 图片相对路径（素材库根目录为基准）
    file_hash: str                    # 图片 MD5 哈希，用于变更检测

    # ========== 核心剧情标签（封闭枚举） ==========
    characters: list[str]             # 画面中出现的角色
    emotions: list[str]               # 角色当前的情绪/心理状态
    events: list[str]                 # 当前画面正在发生的剧情事件

    # ========== 辅助视觉标签 ==========
    description: str = ""             # 整个画面的简洁中文描述（50字以内）
    colors: list[str] = []            # 画面主色调（中文颜色名，2~5个）

    # ========== 追踪信息 ==========
    model: str = ""                   # 打标使用的模型名称
    created_at: str = ""              # 打标时间 ISO 格式
```

### 3.2 封闭枚举候选列表

以下候选列表硬编码在打标 Prompt 中，AI 只能从中选择，不允许创造新值。

**角色列表（characters）：**

| 角色 | 说明 |
|------|------|
| 野原新之助 | 主角，5岁小男孩 |
| 野原美冴 | 小新的妈妈，家庭主妇 |
| 野原广志 | 小新的爸爸，上班族 |
| 野原向日葵 | 小新的妹妹，婴儿 |
| 小白 | 野原家的宠物狗（棉花糖） |
| 风间彻 | 小新的同学，优等生 |
| 樱田妮妮 | 小新的同学，性格强势 |
| 佐藤正男 | 小新的同学，胆小 |
| 阿呆 | 小新的同学，沉默寡言 |
| 酢乙女爱 | 小新的同学，富家女 |
| 松坂老师 | 玫瑰班老师 |
| 吉永老师 | 向日葵班老师 |
| 园长 | 幼稚园园长 |
| 其他 | 不在以上列表中的角色 |

**情绪列表（emotions）：**

| 情绪 | 情绪 | 情绪 |
|------|------|------|
| 开心 | 大笑 | 兴奋 |
| 得意 | 害羞 | 生气 |
| 愤怒 | 哭泣 | 委屈 |
| 害怕 | 紧张 | 震惊 |
| 无语 | 尴尬 | 色眯眯 |
| 搞怪 | 平静 | |

> 如果无法明确判断，默认选择 `平静`。

**事件列表（events）：**

| 类别 | 事件 |
|------|------|
| 家庭日常 | 吃饭、睡觉、洗澡、看电视、聊天、做家务 |
| 搞笑冲突 | 被妈妈骂、恶作剧、捣乱、偷吃零食、逃跑、打闹、吵架 |
| 学校朋友 | 上学、上课、玩耍、比赛 |
| 其他 | 跳舞、唱歌、旅行、冒险、无明显事件 |

> 如果图片是静态人物，没有明显行为，选择 `无明显事件`。

### 3.3 Sidecar 文件格式（`xxx.jpg.tags.json`）

```json
{
  "file_path": "001.jpg",
  "file_hash": "a8f5f167f44f4964e6c998dee827110c",

  "characters": [
    "野原新之助",
    "野原美冴"
  ],
  "emotions": [
    "震惊",
    "害怕",
    "尴尬"
  ],
  "events": [
    "偷吃零食",
    "被妈妈骂"
  ],

  "description": "小新偷吃布丁被美冴发现，在客厅里露出害怕和震惊的表情，额头冒汗，场面十分尴尬。",
  "colors": [
    "黄色",
    "橙色",
    "棕色"
  ],

  "model": "qwen3-vl-flash",
  "created_at": "2026-06-19T15:30:00+08:00"
}
```

---

## 4. Qwen3-VL-Flash 调用设计

### 4.1 接入方式

使用阿里云 DashScope 的 **OpenAI 兼容接口**，与项目现有 `openai` SDK 完全兼容，零额外依赖。

| 配置项 | 值 |
|--------|-----|
| base_url | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| api_key | 复用 `config.toml` 中已有的 `qwen_api_key` |
| model | `qwen3-vl-flash` |
| 调用 SDK | `openai.OpenAI`（项目已有依赖） |
| 图片格式 | base64 data URI：`data:image/jpeg;base64,...` |

### 4.2 调用流程

```
图片文件 → resize（长边≤2048px） → base64编码 → 构造多模态消息 → 调用 OpenAI 兼容接口 → 解析 JSON → 写入 sidecar
```

### 4.3 图片预处理

在发送给 API 之前，对图片进行轻量预处理：

```python
def _prepare_image(image_path: str, max_long_edge: int = 2048) -> str:
    """
    读取图片 → resize → 编码为 base64 data URI。

    - 长边缩放到 max_long_edge（保持宽高比）
    - 转换为 RGB（去除 alpha 通道）
    - JPEG 质量 85%
    - 返回 "data:image/jpeg;base64,xxxx"
    """
```

目的：
- 减少 base64 字符串大小，加快 API 请求速度
- 避免超大图片导致 token 消耗过高
- 统一格式，减少模型处理异常

### 4.4 Prompt 设计

向 Qwen3-VL-Flash 发送如下 System Prompt：

```
你是一名专业的《蜡笔小新》动画素材分析助手。

你的任务是仔细分析输入图片中的角色、情绪、剧情事件和视觉元素，并严格按照指定 JSON 格式返回结果。

要求：

1. characters:
识别图片中出现的主要角色。
允许多个角色。
只能从以下角色列表中选择，不允许创造新的角色名称：

- 野原新之助
- 野原美冴
- 野原广志
- 野原向日葵
- 小白
- 风间彻
- 樱田妮妮
- 佐藤正男
- 阿呆
- 酢乙女爱
- 松坂老师
- 吉永老师
- 园长
- 其他

2. emotions:
分析角色当前表达的情绪和心理状态。
允许多个情绪。
只能从以下情绪列表中选择：

- 开心
- 大笑
- 兴奋
- 得意
- 害羞
- 生气
- 愤怒
- 哭泣
- 委屈
- 害怕
- 紧张
- 震惊
- 无语
- 尴尬
- 色眯眯
- 搞怪
- 平静

如果无法明确判断，选择"平静"。

3. events:
分析当前画面正在发生的剧情事件。
允许多个事件。
只能从以下事件列表中选择：

家庭日常：
- 吃饭
- 睡觉
- 洗澡
- 看电视
- 聊天
- 做家务

搞笑冲突：
- 被妈妈骂
- 恶作剧
- 捣乱
- 偷吃零食
- 逃跑
- 打闹
- 吵架

学校朋友：
- 上学
- 上课
- 玩耍
- 比赛

其他：
- 跳舞
- 唱歌
- 旅行
- 冒险
- 无明显事件

如果图片是静态人物，没有明显行为，则选择"无明显事件"。

4. description:
用一句简洁的中文描述整个画面，长度控制在 50 字以内。
描述中应包含：
- 主要角色
- 主要动作或事件
- 关键情绪

例如：
"小新偷吃布丁被美冴发现，露出震惊和害怕的表情。"

5. colors:
识别画面中的 2~5 个主要颜色。
使用中文颜色名称，例如：
红色、黄色、蓝色、绿色、黑色、白色、橙色、粉色。

返回格式必须严格符合以下 JSON Schema：

{
  "characters": [],
  "emotions": [],
  "events": [],
  "description": "",
  "colors": []
}

重要规则：
- 只能返回 JSON，不允许输出解释、注释或 Markdown。
- 不允许添加 JSON Schema 中不存在的字段。
- 数组必须使用 JSON 数组格式。
- 如果无法识别某项内容，返回空数组或合理默认值。
- 所有文本必须使用简体中文。
```

### 4.5 调用代码（`app/services/llm.py` 新增）

```python
def analyze_image(image_path: str) -> dict:
    """
    调用 Qwen3-VL-Flash (OpenAI 兼容接口) 分析蜡笔小新截图，
    返回 characters / emotions / events / description / colors 字典。
    """
    import base64
    from openai import OpenAI

    # 1. 读取并预处理图片
    image_data_uri = _prepare_image(image_path)

    # 2. 构造 OpenAI 兼容 client
    client = OpenAI(
        api_key=config.app.get("qwen_api_key"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # 3. 构造多模态消息
    messages = [
        {"role": "system", "content": TAGGING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_uri}},
            ],
        },
    ]

    # 4. 调用 API
    response = client.chat.completions.create(
        model="qwen3-vl-flash",
        messages=messages,
        temperature=0.1,        # 低温度保证输出稳定
        max_tokens=500,
    )

    # 5. 解析 JSON（含容错处理）
    raw = response.choices[0].message.content
    return _parse_tags_json(raw)
```

### 4.6 JSON 解析容错

```python
def _parse_tags_json(raw: str) -> dict:
    """
    容错解析 Vision LLM 返回的 JSON。
    处理模型偶尔输出的 Markdown 代码块包裹或多余文本。
    """
    # 尝试直接解析
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 尝试提取 ```json ... ``` 代码块
    match = re.search(r'\{[^{}]*"characters"[^{}]*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # 最终兜底：返回空标签
    logger.error(f"Failed to parse Vision LLM JSON response: {raw[:200]}")
    return {
        "characters": [],
        "emotions": ["平静"],
        "events": ["无明显事件"],
        "description": "",
        "colors": [],
    }
```

### 4.7 成本估算

| 项目 | 数据 |
|------|------|
| Qwen3-VL-Flash 输入价格 | ¥0.001 / 千 tokens |
| Qwen3-VL-Flash 输出价格 | ¥0.002 / 千 tokens |
| 单张图片估算 input | ~500 tokens（base64 压缩后） |
| 单张图片估算 output | ~150 tokens |
| **单张成本** | **约 ¥0.0008** |
| **1000 张成本** | **约 ¥0.80** |
| **10000 张成本** | **约 ¥8.00** |

> Qwen3-VL-Flash 是阿里云性价比最高的视觉模型，百张图成本不到一毛钱。

---

## 5. 核心服务模块设计

### 5.1 `app/services/tagging.py` — 打标核心服务（新建文件）

#### 5.1.1 图片哈希计算

```python
def compute_image_hash(image_path: str) -> str:
    """
    计算图片文件的 MD5 哈希，用于变更检测。

    对大文件（>10MB）采用分块读取，避免内存占用。
    返回 32 位十六进制字符串。
    """
    import hashlib

    hash_md5 = hashlib.md5()
    with open(image_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
```

#### 5.1.2 Sidecar 路径与读写

```python
def get_sidecar_path(image_path: str) -> str:
    """获取 sidecar 标签文件路径"""
    # xxx.jpg → xxx.jpg.tags.json
    return image_path + ".tags.json"


def load_tags(image_path: str) -> Optional[ImageTags]:
    """读取已有标签，文件不存在或损坏返回 None"""
    sidecar = get_sidecar_path(image_path)
    if not os.path.isfile(sidecar):
        return None
    try:
        with open(sidecar, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ImageTags(**data)
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"Corrupted tags file: {sidecar}, error: {e}")
        return None


def save_tags(image_path: str, tags: ImageTags) -> None:
    """写入标签到 sidecar 文件"""
    sidecar = get_sidecar_path(image_path)
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(tags.model_dump(), f, ensure_ascii=False, indent=2)


def delete_tags(image_path: str) -> None:
    """删除 sidecar 标签文件（忽略文件不存在的情况）"""
    sidecar = get_sidecar_path(image_path)
    if os.path.isfile(sidecar):
        os.remove(sidecar)
```

#### 5.1.3 需要打标的图片发现

```python
def find_images_needing_tags(
    base_dir: str,
    force: bool = False,
) -> List[str]:
    """
    扫描素材目录，返回需要打标的图片绝对路径列表。

    判断逻辑：
    - force=True：忽略已有标签，全部重新打标
    - force=False：
      · 不存在 sidecar 文件 → 需要打标
      · 存在 sidecar 但 file_hash 与当前文件 MD5 不匹配 → 需要打标
      · 存在 sidecar 且 hash 匹配 → 跳过
    """
    allowed_ext = (".jpg", ".jpeg", ".png", ".bmp")
    images = []

    for root, dirs, files in os.walk(base_dir):
        # 仅扫描一级子目录，与现有素材扫描逻辑保持一致
        for f in files:
            if f.lower().endswith(allowed_ext):
                images.append(os.path.join(root, f))

    if force:
        return images

    needing = []
    for img_path in images:
        existing = load_tags(img_path)
        if existing is None:
            needing.append(img_path)
        elif existing.file_hash != compute_image_hash(img_path):
            needing.append(img_path)
        # else: hash 匹配，跳过

    return needing
```

#### 5.1.4 单张图片打标

```python
def tag_single_image(
    image_path: str,
    base_dir: str,
) -> ImageTags:
    """
    对单张图片执行 AI 打标。

    流程：
    1. 计算文件哈希
    2. 调用 llm.analyze_image() 获取标签
    3. 填充 ImageTags 模型（file_path 为相对于 base_dir 的路径）
    4. 写入 sidecar 文件
    5. 返回 ImageTags 对象
    """
    file_hash = compute_image_hash(image_path)
    rel_path = os.path.relpath(image_path, base_dir)

    result = llm.analyze_image(image_path)

    tags = ImageTags(
        file_path=rel_path,
        file_hash=file_hash,
        characters=result.get("characters", []),
        emotions=result.get("emotions", ["平静"]),
        events=result.get("events", ["无明显事件"]),
        description=result.get("description", ""),
        colors=result.get("colors", []),
        model="qwen3-vl-flash",
        created_at=datetime.now().isoformat(),
    )

    save_tags(image_path, tags)
    return tags
```

#### 5.1.5 批量打标调度

```python
def batch_tag_images(
    base_dir: str,
    force: bool = False,
    max_concurrent: int = 3,
    task_id: str = None,
) -> dict:
    """
    批量对素材库中的图片进行 AI 打标。

    返回: {"total": N, "tagged": N, "skipped": N, "failed": N, "errors": [...]}

    实现要点：
    - 使用 concurrent.futures.ThreadPoolExecutor 控制并发
    - 每个图片独立调用 Vision API，单张失败不影响其他图片
    - 通过 task_id 向任务系统上报进度
    - API 调用之间间隔 0.3-0.5s，避免触发 DashScope rate limit
    """
    images = find_images_needing_tags(base_dir, force=force)
    total = len(images)
    tagged = 0
    skipped = 0
    failed = 0
    errors = []

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {
            executor.submit(tag_single_image, img, base_dir): img
            for img in images
        }
        for future in as_completed(futures):
            img = futures[future]
            try:
                future.result()
                tagged += 1
            except Exception as e:
                failed += 1
                errors.append({"file": img, "error": str(e)})
                logger.error(f"Failed to tag {img}: {e}")

            # 更新进度
            if task_id:
                _update_progress(task_id, tagged + skipped + failed, total)

            # API 限速间隔
            time.sleep(0.3)

    return {
        "total": total,
        "tagged": tagged,
        "skipped": skipped,
        "failed": failed,
        "errors": errors,
    }
```

#### 5.1.6 标签搜索

```python
def search_materials_by_tags(
    base_dir: str,
    characters: List[str] = None,
    emotions: List[str] = None,
    events: List[str] = None,
    keyword: str = None,
    match_mode: str = "any",
) -> List[dict]:
    """
    根据标签搜索匹配的图片素材。

    搜索维度：
    - characters: 按角色精确匹配（从封闭枚举中选择）
    - emotions: 按情绪精确匹配（从封闭枚举中选择）
    - events: 按事件精确匹配（从封闭枚举中选择）
    - keyword: 在 description 中模糊搜索

    match_mode:
    - "any": 任意维度命中即匹配
    - "all": 所有指定维度都必须命中

    返回：匹配的素材列表，含文件路径、标签内容、匹配得分。

    匹配得分计算：
    - 每个命中的 character +1
    - 每个命中的 emotion +1
    - 每个命中的 event +1
    - keyword 命中 description +2（语义匹配权重更高）
    """
    results = []
    all_tags = _load_all_tags(base_dir)

    for tags in all_tags:
        score = 0
        match_detail = {
            "characters_matched": [],
            "emotions_matched": [],
            "events_matched": [],
            "keyword_matched": False,
        }

        # 角色匹配
        if characters:
            matched_chars = [c for c in characters if c in tags.characters]
            score += len(matched_chars)
            match_detail["characters_matched"] = matched_chars

        # 情绪匹配
        if emotions:
            matched_emotions = [e for e in emotions if e in tags.emotions]
            score += len(matched_emotions)
            match_detail["emotions_matched"] = matched_emotions

        # 事件匹配
        if events:
            matched_events = [e for e in events if e in tags.events]
            score += len(matched_events)
            match_detail["events_matched"] = matched_events

        # 描述关键词匹配
        if keyword and keyword in tags.description:
            score += 2
            match_detail["keyword_matched"] = True

        if score > 0:
            results.append({
                "file_path": tags.file_path,
                "characters": tags.characters,
                "emotions": tags.emotions,
                "events": tags.events,
                "description": tags.description,
                "colors": tags.colors,
                "match_score": score,
                "match_detail": match_detail,
            })

    # 按得分降序
    results.sort(key=lambda r: r["match_score"], reverse=True)
    return results
```

#### 5.1.7 标签统计

```python
def get_tag_statistics(base_dir: str) -> dict:
    """
    获取素材库的标签统计信息。

    返回：
    {
        "total_images": 150,
        "tagged_count": 120,
        "untagged_count": 30,
        "character_distribution": {"野原新之助": 85, "野原美冴": 40, ...},
        "emotion_distribution": {"开心": 55, "生气": 30, ...},
        "event_distribution": {"吃饭": 20, "被妈妈骂": 15, ...},
        "color_distribution": {"黄色": 60, "蓝色": 35, ...},
        "avg_tags_per_image": 5.2,
    }
    """
    all_tags = _load_all_tags(base_dir)
    all_images = _find_all_images(base_dir)

    total = len(all_images)
    tagged = len(all_tags)

    char_dist = {}
    emo_dist = {}
    evt_dist = {}
    col_dist = {}

    for tags in all_tags:
        for c in tags.characters:
            char_dist[c] = char_dist.get(c, 0) + 1
        for e in tags.emotions:
            emo_dist[e] = emo_dist.get(e, 0) + 1
        for ev in tags.events:
            evt_dist[ev] = evt_dist.get(ev, 0) + 1
        for co in tags.colors:
            col_dist[co] = col_dist.get(co, 0) + 1

    # 按频次降序
    char_dist = dict(sorted(char_dist.items(), key=lambda x: x[1], reverse=True))
    emo_dist = dict(sorted(emo_dist.items(), key=lambda x: x[1], reverse=True))
    evt_dist = dict(sorted(evt_dist.items(), key=lambda x: x[1], reverse=True))
    col_dist = dict(sorted(col_dist.items(), key=lambda x: x[1], reverse=True))

    total_tags = sum(len(t.characters) + len(t.emotions) + len(t.events) for t in all_tags)

    return {
        "total_images": total,
        "tagged_count": tagged,
        "untagged_count": total - tagged,
        "character_distribution": char_dist,
        "emotion_distribution": emo_dist,
        "event_distribution": evt_dist,
        "color_distribution": col_dist,
        "avg_tags_per_image": round(total_tags / tagged, 1) if tagged > 0 else 0,
    }
```

### 5.2 `app/services/video.py` 扩展 — 基于标签的素材匹配

在素材预处理流程中增加基于标签的匹配能力。当用户开启 `match_materials_to_script` 时，利用脚本内容匹配最相关的已打标图片：

```python
def match_materials_by_tags(
    script_text: str,
    tagged_dir: str,
    top_k: int = 5,
) -> List[str]:
    """
    给定视频脚本，自动匹配与内容最相关的打标图片。

    实现思路：
    1. 调用 LLM 分析脚本：这段脚本涉及哪些角色、什么情绪、什么场景事件？
       返回与 ImageTags 同 schema 的 dict（{characters, emotions, events, keyword}）
    2. 调用 search_materials_by_tags() 搜索匹配素材
    3. 按 match_score 降序返回 top_k 个素材路径

    示例：
    脚本提到"小新偷吃零食被妈妈发现"
    → LLM 分析 → characters: ["野原新之助", "野原美冴"], events: ["偷吃零食", "被妈妈骂"]
    → 搜索 → 返回最匹配的 N 张图片
    """
```

实现注意事项：
- 脚本分析使用**同一个 LLM**（用户配置的 `llm_provider`），不需要 Vision 能力，仅做文本→标签映射
- 如果素材库中无匹配的已打标图片，fallback 到现有的随机选择逻辑
- 仅在 `video_source == "local"` 且素材库存在打标文件时生效

---

## 6. API 接口设计

所有接口注册在 `app/controllers/v1/video.py` 的 router 下，路径前缀 `/api/v1/materials`。

### 6.1 `GET /api/v1/materials/tags` — 标签统计概览

**Response:**
```json
{
  "status": 200,
  "message": "success",
  "data": {
    "total_images": 150,
    "tagged_count": 120,
    "untagged_count": 30,
    "character_distribution": {
      "野原新之助": 85,
      "野原美冴": 40,
      "野原广志": 30,
      "风间彻": 25
    },
    "emotion_distribution": {
      "开心": 55,
      "生气": 30,
      "震惊": 20,
      "尴尬": 18
    },
    "event_distribution": {
      "吃饭": 20,
      "被妈妈骂": 15,
      "玩耍": 12,
      "无明显事件": 35
    },
    "color_distribution": {
      "黄色": 60,
      "蓝色": 35,
      "白色": 30
    },
    "avg_tags_per_image": 5.2
  }
}
```

### 6.2 `POST /api/v1/materials/tags/generate` — 触发批量打标

**Request Body:**
```json
{
  "force": false,
  "max_concurrent": 3
}
```

**Response:**
```json
{
  "status": 200,
  "message": "Tagging task started",
  "data": {
    "task_id": "a1b2c3d4-..."
  }
}
```

参数说明：
- `force`：`false` 仅处理未打标/已变更的图片；`true` 强制全部重新打标
- `max_concurrent`：并发数，建议 2~5，默认 3
- `directory` 不需要传，自动使用 `config.toml` 中配置的 `material_directory`

### 6.3 `GET /api/v1/materials/tags/status?task_id=xxx` — 查询打标进度

**Response:**
```json
{
  "status": 200,
  "message": "success",
  "data": {
    "task_id": "a1b2c3d4-...",
    "state": "running",
    "progress": 45,
    "total": 120,
    "tagged": 54,
    "skipped": 0,
    "failed": 0,
    "current_file": "001.jpg",
    "errors": []
  }
}
```

状态枚举：`pending` → `running` → `completed` / `failed` / `cancelled`

### 6.4 `GET /api/v1/materials/tags/search` — 按标签搜索素材

**Query Parameters:**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `characters` | string | 否 | 逗号分隔的角色名，如 `野原新之助,野原美冴` |
| `emotions` | string | 否 | 逗号分隔的情绪名，如 `震惊,害怕` |
| `events` | string | 否 | 逗号分隔的事件名，如 `偷吃零食,被妈妈骂` |
| `keyword` | string | 否 | 描述关键词模糊搜索 |
| `match` | string | 否 | `any`（默认）或 `all` |
| `limit` | int | 否 | 返回条数上限，默认 20 |

**示例请求：**
```
GET /api/v1/materials/tags/search?characters=野原新之助,野原美冴&events=偷吃零食&match=any&limit=10
```

**Response:**
```json
{
  "status": 200,
  "message": "success",
  "data": {
    "results": [
      {
        "file_path": "001.jpg",
        "characters": ["野原新之助", "野原美冴"],
        "emotions": ["震惊", "害怕"],
        "events": ["偷吃零食", "被妈妈骂"],
        "description": "小新偷吃布丁被美冴发现，露出害怕和震惊的表情。",
        "colors": ["黄色", "橙色"],
        "match_score": 4,
        "match_detail": {
          "characters_matched": ["野原新之助", "野原美冴"],
          "emotions_matched": [],
          "events_matched": ["偷吃零食"],
          "keyword_matched": false
        }
      }
    ],
    "total": 5
  }
}
```

### 6.5 `DELETE /api/v1/materials/tags` — 删除标签

**Request Body:**
```json
{
  "file_paths": ["001.jpg", "002.jpg"]
}
```

**Response:**
```json
{
  "status": 200,
  "message": "success",
  "data": {
    "deleted": 2,
    "not_found": 0
  }
}
```

### 6.6 `GET /api/v1/materials/tags/{file_path}` — 获取单张图片标签

**Response:**
```json
{
  "status": 200,
  "message": "success",
  "data": {
    "file_path": "001.jpg",
    "file_hash": "a8f5f167f44f4964e6c998dee827110c",
    "characters": ["野原新之助", "野原美冴"],
    "emotions": ["震惊", "害怕", "尴尬"],
    "events": ["偷吃零食", "被妈妈骂"],
    "description": "小新偷吃布丁被美冴发现...",
    "colors": ["黄色", "橙色", "棕色"],
    "model": "qwen3-vl-flash",
    "created_at": "2026-06-19T15:30:00+08:00"
  }
}
```

---

## 7. WebUI 设计

### 7.1 素材选择区域增强（`webui/Main.py` — 现有素材管理区域）

在现有 `get_material_files()` 函数上方的位置，增加筛选工具栏：

```
┌─────────────────────────────────────────────────────────┐
│  🔍 筛选素材                                             │
│  ┌────────────┬────────────┬────────────┬────────────┐  │
│  │ 角色 ▼     │ 情绪 ▼     │ 事件 ▼     │ 搜索...    │  │
│  │ ☑ 小新    │ ☑ 开心    │ ☑ 被骂    │            │  │
│  │ ☐ 美冴    │ ☐ 生气    │ ☐ 吃饭    │            │  │
│  │ ☐ 广志    │ ☐ 震惊    │ ☐ 玩耍    │            │  │
│  └────────────┴────────────┴────────────┴────────────┘  │
│                                                         │
│  匹配结果: 5 张  │  排序: [匹配度 ▼]                     │
│                                                         │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐                   │
│  │ [缩略图] │ │ [缩略图] │ │ [缩略图] │                   │
│  │ 001.jpg │ │ 003.jpg │ │ 007.jpg │                   │
│  │ 👤小新   │ │ 👤小新   │ │ 👤小新   │                   │
│  │ 😨震惊   │ │ 😡生气   │ │ 😂大笑   │                   │
│  │ 🏷被骂   │ │ 🏷捣乱   │ │ 🏷玩耍   │                   │
│  │ ★★★★   │ │ ★★★    │ │ ★★     │                   │
│  └─────────┘ └─────────┘ └─────────┘                   │
│                                                         │
│  点击卡片 → 弹窗查看完整标签 + 大图预览                    │
└─────────────────────────────────────────────────────────┘
```

实现要点：
- 角色/情绪/事件筛选下拉框使用 `st.multiselect`，选项从已打标数据中动态获取（而非硬编码枚举列表）
- 搜索框支持在 description 中模糊匹配
- 筛选条件变更时实时刷新结果（无需点按钮）
- 已打标图片显示标签 chip（`tag_character` / `tag_emotion` / `tag_event`），未打标图片显示灰色「未打标」标记
- 每个素材卡片显示 match_score 星级

### 7.2 新增「AI 打标」管理页面（`webui/pages/tagging.py` 新建）

在 WebUI 侧边栏新增「🤖 AI 打标」入口，路由到独立 Streamlit 页面：

```
┌─────────────────────────────────────────────────────────┐
│  🤖 AI 图片打标管理                                      │
│                                                         │
│  ── 📊 概览 ──────────────────────────────────────────  │
│  ┌──────────┬──────────┬──────────┬──────────┐         │
│  │ 总图片   │ 已打标   │ 未打标   │ 平均标签  │         │
│  │   150    │   120    │    30    │   5.2    │         │
│  └──────────┴──────────┴──────────┴──────────┘         │
│                                                         │
│  ── 🔧 打标设置 ──────────────────────────────────────  │
│  模型: Qwen3-VL-Flash  │  并发数: [ 3 ▼ ]              │
│  ○ 仅处理未打标图片（推荐）                              │
│  ○ 强制重新打标所有图片                                  │
│                                                         │
│  [🚀 开始打标]  [⏹ 取消]                               │
│                                                         │
│  ── 📈 进度 ──────────────────────────────────────────  │
│  ████████████████████░░░░░░░  75% (90/120)              │
│  当前: 001.jpg                                          │
│  成功: 85  │  跳过: 20  │  失败: 2                      │
│                                                         │
│  ── 🏷 角色分布 ──────────────────────────────────────  │
│  野原新之助  ████████████████████████  85               │
│  野原美冴    ██████████████  40                          │
│  野原广志    ██████████  30                              │
│  风间彻      ████████  25                                │
│  樱田妮妮    ██████  18                                  │
│  ...                                                    │
│                                                         │
│  ── 😨 情绪分布 ──────────────────────────────────────  │
│  开心  ██████████████████  55                            │
│  生气  ██████████  30                                    │
│  震惊  ██████  20                                        │
│  尴尬  ██████  18                                        │
│  ...                                                    │
│                                                         │
│  ── 🎬 事件分布 ──────────────────────────────────────  │
│  无明显事件  ██████████████  35                           │
│  吃饭  ██████  20                                        │
│  被妈妈骂  █████  15                                     │
│  玩耍  ████  12                                          │
│  ...                                                    │
│                                                         │
│  ── 🔍 快速搜索 ──────────────────────────────────────  │
│  角色: [野原新之助 ▼]  情绪: [任意 ▼]  事件: [任意 ▼]    │
│  描述关键词: [____________]                              │
│  [🔍 搜索]                                              │
│  （搜索结果以卡片网格展示，同 7.1 的卡片样式）             │
└─────────────────────────────────────────────────────────┘
```

实现要点：
- 打标进度通过轮询 `GET /api/v1/materials/tags/status` 实现（每 2 秒）
- 分布图使用 `st.bar_chart` 或 `st.dataframe` 渲染
- 统计数据和分布图仅在打标任务完成后刷新
- 搜索区域的角色下拉框选项从实际数据动态获取

### 7.3 侧边栏导航

在现有 Streamlit 侧边栏增加导航项：

```python
# webui/Main.py 侧边栏新增
page = st.sidebar.radio(
    "页面导航",
    ["🎬 视频生成", "🤖 AI 打标"],
)
if page == "🤖 AI 打标":
    import webui.pages.tagging as tagging
    tagging.render()
```

### 7.4 详情弹窗

点击搜索结果中的任意素材卡片，展示完整标签信息：

```
┌──────────────────────────────────────┐
│  素材详情: 001.jpg           [✕ 关闭]│
│                                      │
│  ┌────────────────────────────────┐  │
│  │       [大图预览区域]            │  │
│  │                                │  │
│  └────────────────────────────────┘  │
│                                      │
│  👤 角色:  野原新之助, 野原美冴       │
│  😨 情绪:  震惊, 害怕, 尴尬          │
│  🎬 事件:  偷吃零食, 被妈妈骂         │
│  📝 描述:  小新偷吃布丁被美冴发现，   │
│            露出害怕和震惊的表情。      │
│  🎨 颜色:  黄色, 橙色, 棕色          │
│                                      │
│  模型: qwen3-vl-flash                │
│  打标时间: 2026-06-19 15:30          │
│                                      │
│  [🗑 重新打标]  [✏ 手动编辑]         │
└──────────────────────────────────────┘
```

---

## 8. 配置文件扩展

### 8.1 `config.toml` 新增 `[tagging]` 配置段

```toml
[tagging]
# 是否启用 AI 打标功能
enabled = true

# ====== 模型配置 ======

# Vision 模型名称（DashScope OpenAI 兼容接口）
vision_model = "qwen3-vl-flash"

# DashScope OpenAI 兼容接口 base_url
vision_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# API key 复用 qwen_api_key（在 [app] 段中配置），无需单独设置
# 如果将来需要单独配置打标 API key，可在此覆盖：
vision_api_key_override = ""

# ====== 打标行为配置 ======

# 批量打标时的并发数（控制 API 并发）
max_concurrent = 3

# API 调用间隔（秒），避免触发 rate limit
request_interval = 0.3

# 标签缓存有效期（天），超过后自动重新打标。0 表示永不过期（仅靠哈希校验）
cache_ttl_days = 0

# 图片上传后是否自动打标
auto_tag_on_upload = false

# ====== 图片预处理 ======

# 发送给 API 前的图片长边最大像素（减少 base64 大小）
max_image_long_edge = 2048

# 跳过超大文件（MB），超过此阈值的图片不进行打标
max_file_size_mb = 20
```

### 8.2 配置加载（`app/config/config.py` 扩展）

复用现有的 `config.load_config()` 机制（基于 `toml` 库读取 `config.toml`）。`[tagging]` 段的值通过 `config.tagging.get("key")` 访问，与现有 `config.app`、`config.ui` 模式一致。

---

## 9. 实施步骤

### Phase 1：基础能力（预计 2 天）

| # | 任务 | 涉及文件 | 说明 |
|---|------|---------|------|
| 1.1 | 定义数据模型 | `app/models/schema.py` | 新增 `ImageTags(BaseModel)`，包含 characters / emotions / events / description / colors / file_path / file_hash / model / created_at |
| 1.2 | 实现图片分析函数 | `app/services/llm.py` | 新增 `analyze_image(image_path) → dict`：调用 Qwen3-VL-Flash（OpenAI 兼容接口），包含图片预处理 (`_prepare_image`)、Prompt 常量 (`TAGGING_SYSTEM_PROMPT`)、JSON 解析容错 (`_parse_tags_json`) |
| 1.3 | 实现打标核心服务 | `app/services/tagging.py`（新建） | `compute_image_hash()`、`load_tags()` / `save_tags()` / `delete_tags()`、`tag_single_image()`、`find_images_needing_tags()` |
| 1.4 | 添加配置项 | `app/config/config.py` + `config.toml` | `[tagging]` 配置段，读取逻辑与现有 config 保持一致 |

### Phase 2：后端 API（预计 2 天）

| # | 任务 | 涉及文件 | 说明 |
|---|------|---------|------|
| 2.1 | 标签管理接口 | `app/controllers/v1/video.py` | 6 个 REST 端点：GET stats / POST generate / GET status / GET search / DELETE tags / GET single tags |
| 2.2 | 批量打标任务调度 | `app/services/tagging.py` | `batch_tag_images()`：ThreadPoolExecutor + 进度上报 + 任务取消支持 |
| 2.3 | 标签搜索实现 | `app/services/tagging.py` | `search_materials_by_tags()`：characters/emotions/events 精确匹配 + description 模糊搜索 + 匹配得分 |
| 2.4 | 进度追踪集成 | `app/services/tagging.py` + `app/services/state.py` | 复用 `MemoryState` 追踪打标任务状态，与现有视频生成任务保持一致 |

### Phase 3：WebUI（预计 2-3 天）

| # | 任务 | 涉及文件 | 说明 |
|---|------|---------|------|
| 3.1 | 打标管理页面 | `webui/pages/tagging.py`（新建） | 概览统计卡片 + 打标配置 + 进度条 + 角色/情绪/事件分布柱状图 + 快速搜索 |
| 3.2 | 素材选择增强 | `webui/Main.py` | 角色/情绪/事件多选筛选器 + 关键词搜索框 + 筛选结果卡片网格（含标签 chip + 匹配度星级） |
| 3.3 | 详情弹窗 | `webui/Main.py` 或 `webui/components/tag_detail.py` | 点击卡片 → 大图预览 + 完整标签信息 + 重新打标按钮 |
| 3.4 | 侧边栏导航 | `webui/Main.py` | `st.sidebar.radio` 添加「🤖 AI 打标」页面切换 |

### Phase 4：集成与测试（预计 1-2 天）

| # | 任务 | 涉及文件 | 说明 |
|---|------|---------|------|
| 4.1 | 标签匹配素材 | `app/services/video.py` + `app/services/task.py` | 当 `match_materials_to_script=True` 时，调用 LLM 将脚本 → 标签 → 搜索最优匹配图片 |
| 4.2 | CLI 支持 | `cli.py` | `python cli.py tag-images`：命令行触发批量打标，输出进度到终端 |
| 4.3 | 国际化文本 | `webui/i18n/zh.json` 等 | 新增打标相关的 UI 文本 key |
| 4.4 | 单元测试 | `test/services/test_tagging.py`（新建） | compute_image_hash、load/save/delete_tags 的单元测试 + search 功能测试 |
| 4.5 | 端到端验证 | 手动测试 | 准备 10-20 张蜡笔小新截图 → 打标 → 验证标签准确性 → 搜索筛选 → 视频生成素材匹配 |

---

## 10. 错误处理策略

| 异常场景 | 处理方式 |
|---------|---------|
| 图片文件损坏/无法读取 | 捕获 PIL `UnidentifiedImageError`，记录错误，跳过该图片 |
| API 网络超时 | `openai` SDK 自动重试 2 次；仍失败则标记该图片为 `failed`，继续处理下一张 |
| API 返回非 JSON | `_parse_tags_json()` 尝试正则提取 JSON 块；失败则返回默认空标签 `{"characters":[], "emotions":["平静"], "events":["无明显事件"], "description":"", "colors":[]}` |
| API Rate Limit (429) | 自动退避：等待 3s 重试，仍 429 则等待 10s 重试，第 3 次仍失败则跳过 |
| API 认证失败 (401/403) | 立即终止批量任务，向前端返回明确错误信息：「API Key 无效，请检查 config.toml 中的 qwen_api_key」 |
| 图片过大（>20MB） | 预处理阶段检查文件大小，超过阈值跳过并记录 |
| sidecar 文件写入失败 | 捕获 `IOError`，记录错误日志，不影响其他图片处理 |
| sidecar 文件 JSON 损坏 | `load_tags()` 返回 `None`，自动触发重新打标 |
| 批量任务被用户取消 | `batch_tag_images()` 检查 `task_id` 对应的任务状态，若被取消则停止提交新任务并清理 |

---

## 11. 依赖分析

**零新增第三方依赖。** 所有能力来自项目已有的包：

| 能力 | 依赖 | 来源 |
|------|------|------|
| Vision API 调用 | `openai` | 已有（`pyproject.toml`） |
| 图片读写 / resize | `PIL` (Pillow) | 已有（moviepy 传递依赖） |
| 数据模型 | `pydantic` | 已有 |
| JSON 序列化 | `json`（标准库） | Python 内置 |
| MD5 哈希 | `hashlib`（标准库） | Python 内置 |
| 并发控制 | `concurrent.futures`（标准库） | Python 内置 |
| 文件系统操作 | `os` / `glob`（标准库） | Python 内置 |
| 日志 | `loguru` | 已有 |
| 配置读取 | `toml` | 已有（传递依赖） |

---

## 12. 可选扩展方向

| 方向 | 描述 | 优先级 |
|------|------|--------|
| 批量手动修正 | 发现 AI 标错的图片，支持在 WebUI 中批量修改标签 | 高 |
| 视频首帧打标 | 对本地视频素材的首帧/关键帧进行同样的打标 | 中 |
| 正片场景分类 | 根据剧情阶段（日常/冲突/解决）自动为素材分组 | 中 |
| 连续帧去重 | 检测高度相似的连续截图，避免重复打标 | 低 |
| 标签导出 | 导出为 CSV / JSON 供外部工具使用 | 低 |
| 自定义 Prompt | 允许用户在 WebUI 中修改角色/情绪/事件候选列表 | 低 |
| 向量相似度 | 使用 CLIP embedding 做视觉相似度搜索（超越标签的语义匹配） | 低 |
