from pathlib import Path
import base64
import json
import os
import uuid
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from openai import OpenAI
from pydantic import BaseModel, Field

import psycopg
from psycopg.rows import dict_row


# ============================================================
# 基础配置
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# 本地开发时读取 .env
# Vercel 上没有 .env 也不会报错
load_dotenv(BASE_DIR / ".env")

app = FastAPI(
    title="AIGC MFT 公益广告实验平台"
)


# ============================================================
# PostgreSQL
# ============================================================

DATABASE_URL = os.getenv("DATABASE_URL")


def db():
    """
    PostgreSQL 数据库连接。

    Vercel:
        使用 Vercel / Neon 的 DATABASE_URL。

    本地:
        可以在项目根目录 .env 中配置 DATABASE_URL。
    """

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError(
            "未配置 DATABASE_URL。"
            "请在 Vercel Environment Variables 中配置。"
        )

    return psycopg.connect(
        database_url,
        row_factory=dict_row,
        connect_timeout=10
    )


def init_db():
    """
    初始化数据库表。

    注意：
    不在模块 import 阶段强制执行。
    Vercel Serverless 环境中由请求触发初始化。
    """

    conn = db()

    try:

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                topic TEXT NOT NULL,
                care DOUBLE PRECISION NOT NULL,
                fairness DOUBLE PRECISION NOT NULL,
                loyalty DOUBLE PRECISION NOT NULL,
                authority DOUBLE PRECISION NOT NULL,
                sanctity DOUBLE PRECISION NOT NULL,
                matched_foundation TEXT NOT NULL,
                unmatched_foundation TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ads (
                id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                condition TEXT NOT NULL,
                foundation TEXT NOT NULL,
                strategy_json TEXT NOT NULL,
                copy TEXT NOT NULL,
                image_prompt TEXT NOT NULL,
                image_url TEXT,
                FOREIGN KEY(experiment_id)
                    REFERENCES experiments(id)
                    ON DELETE CASCADE
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluations (
                id BIGSERIAL PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                ad_id TEXT NOT NULL,
                moral_resonance INTEGER NOT NULL,
                emotional_response INTEGER NOT NULL,
                persuasion INTEGER NOT NULL,
                behavioral_intention INTEGER NOT NULL,
                created_at TIMESTAMPTZ NOT NULL,
                FOREIGN KEY(experiment_id)
                    REFERENCES experiments(id)
                    ON DELETE CASCADE,
                FOREIGN KEY(ad_id)
                    REFERENCES ads(id)
                    ON DELETE CASCADE
            )
            """
        )

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


# ============================================================
# MFT 五大道德基础
# ============================================================

FOUNDATIONS = {
    "care": {
        "label": "Care / Harm",
        "cn": "关怀 / 伤害"
    },
    "fairness": {
        "label": "Fairness / Cheating",
        "cn": "公平 / 欺骗"
    },
    "loyalty": {
        "label": "Loyalty / Betrayal",
        "cn": "忠诚 / 背叛"
    },
    "authority": {
        "label": "Authority / Subversion",
        "cn": "权威 / 颠覆"
    },
    "sanctity": {
        "label": "Sanctity / Degradation",
        "cn": "神圣 / 堕落"
    }
}


# ============================================================
# Pydantic
# ============================================================

class MFTScores(BaseModel):

    care: float = Field(ge=0, le=10)
    fairness: float = Field(ge=0, le=10)
    loyalty: float = Field(ge=0, le=10)
    authority: float = Field(ge=0, le=10)
    sanctity: float = Field(ge=0, le=10)


class GenerateRequest(BaseModel):

    topic: str = Field(
        min_length=2,
        max_length=200
    )

    mft: MFTScores


class EvaluationRequest(BaseModel):

    experiment_id: str

    ad_id: str

    moral_resonance: int = Field(
        ge=1,
        le=7
    )

    emotional_response: int = Field(
        ge=1,
        le=7
    )

    persuasion: int = Field(
        ge=1,
        le=7
    )

    behavioral_intention: int = Field(
        ge=1,
        le=7
    )


# ============================================================
# API Client
# ============================================================

def get_client(
    api_key_name: str,
    base_url_name: str
):

    key = os.getenv(api_key_name)
    base_url = os.getenv(base_url_name)

    if not key:
        raise HTTPException(
            status_code=500,
            detail=(
                f"未配置 {api_key_name}。"
                "请在 Vercel Environment Variables 中配置。"
            )
        )

    kwargs = {
        "api_key": key
    }

    if base_url:
        kwargs["base_url"] = base_url

    return OpenAI(**kwargs)


# ============================================================
# MFT
# ============================================================

def scores_dict(mft: MFTScores):

    return {
        "care": mft.care,
        "fairness": mft.fairness,
        "loyalty": mft.loyalty,
        "authority": mft.authority,
        "sanctity": mft.sanctity
    }


def choose_conditions(mft: MFTScores):

    scores = scores_dict(mft)

    ordered = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return (
        ordered[0][0],
        ordered[-1][0]
    )


# ============================================================
# LLM Prompt
# ============================================================

def build_generation_prompt(
    topic: str,
    mft: MFTScores,
    matched: str,
    unmatched: str
):

    scores = scores_dict(mft)

    foundation_desc = "\n".join(
        f"- {k}: "
        f"{FOUNDATIONS[k]['label']} / "
        f"{FOUNDATIONS[k]['cn']} = {v}"
        for k, v in scores.items()
    )

    return f"""
你是“个性化公益广告实验”的高级创意总监、视觉设计师和MFT实验设计专家。

研究目标：

验证“受众的MFT道德基础与公益广告诉求匹配时，是否比不匹配时产生更强的道德共鸣、情感反应、说服效果和行为意愿”。

公益主题：

{topic}

受众MFT五维人工设定分数：

{foundation_desc}

匹配条件：

{matched}
{FOUNDATIONS[matched]['label']}
{FOUNDATIONS[matched]['cn']}

不匹配条件：

{unmatched}
{FOUNDATIONS[unmatched]['label']}
{FOUNDATIONS[unmatched]['cn']}

这不是普通的两张公益图片。

必须设计成：

“同一个公益主题 + 两种明显不同的MFT道德诉求框架”。

两个版本必须让普通受试者一眼看出：

它们讨论的是同一个公益问题，
但是价值诉求、视觉隐喻、传播逻辑和视觉叙事明显不同。

禁止：

“同一个主体 + 同一个场景 + 同一个构图 + 只改变几个关键词”。

允许并鼓励：

- 使用完全不同的主体
- 使用完全不同的场景
- 使用完全不同的视觉隐喻
- 使用完全不同的构图
- 使用完全不同的视觉符号
- 使用完全不同的叙事方式

但是：

两张海报必须拥有相同的公益主题。

匹配版必须围绕：

{FOUNDATIONS[matched]['cn']}

展开。

必须让普通受试者能够感受到：

“这张公益广告正在呼吁我保护、维护或避免伤害我最重视的道德价值。”

不匹配版必须围绕：

{FOUNDATIONS[unmatched]['cn']}

展开。

但是视觉概念必须与匹配版明显不同。

必须尽可能改变：

- 核心主体
- 场景
- 视觉叙事
- 视觉隐喻
- 构图重心
- 摄影角度
- 情绪
- 视觉符号
- 色彩关系
- 文字排版风格

必须生成两句中文公益广告标语。

两句标语：

1. 必须围绕完全相同的公益主题。
2. 必须形成明显的对仗关系。
3. 句式长度尽量接近。
4. 结构尽量对应。
5. 不能使用学术术语。
6. 必须自然，像真正的公益广告。
7. 两句必须体现不同MFT道德价值。
8. 必须具有传播性和记忆点。

这是一个海报生成任务。

图片中必须直接出现对应的中文公益广告标语。

禁止把标语留给网页后期叠加。

image_prompt必须明确写出：

EXACT CHINESE SLOGAN TO RENDER:

“这里放copy中的完整中文标语”

图像模型必须尝试将这句话：

逐字、完整、清晰地生成在海报中。

不得修改文字。
不得增加额外文字。
不得删除文字。
不得生成英文翻译。
不得生成乱码。
不得生成Logo。
不得生成水印。

必须生成：

2:3 vertical public service advertising poster.

尺寸：

1024x1536。

必须是完整的公益广告海报。

必须具有：

- 明确视觉主体
- 明确视觉焦点
- 前景、中景、背景层次
- 专业摄影或广告艺术设计
- 强烈视觉隐喻
- 合理文字区域
- 高质量商业广告完成度

标语可以位于：

- 上方
- 中央
- 下方
- 左侧
- 右侧
- 与主体融合的位置

位置不限。

但必须：

清晰、完整、可读。

允许：

- 对应的中文公益广告标语

禁止：

- Logo
- 水印
- 品牌名称
- 网址
- 其他无关文字
- 随机英文
- 随机字母
- 额外标语

匹配版和不匹配版必须：

视觉概念明显不同。

不要只是改变颜色。

不要只是改变文字。

不要只是改变一个主体。

必须至少在以下六个方面发生明显变化：

1. 核心主体
2. 场景
3. 视觉隐喻
4. 构图
5. 情绪
6. 视觉符号

但是：

两张海报的摄影质量、分辨率、完成度必须保持同等级。

不能让匹配版天然更漂亮。

不要Markdown。

不要代码块。

只输出合法JSON。

结构严格为：

{{
"matched": {{
"foundation": "{matched}",
"strategy": {{
"core_value": "...",
"emotion": ["...", "..."],
"narrative": "...",
"visual_metaphor": "...",
"visual_subject": "...",
"composition": "...",
"lighting": "...",
"typography_style": "..."
}},
"copy": "...",
"image_prompt": "..."
}},

"unmatched": {{
"foundation": "{unmatched}",
"strategy": {{
"core_value": "...",
"emotion": ["...", "..."],
"narrative": "...",
"visual_metaphor": "...",
"visual_subject": "...",
"composition": "...",
"lighting": "...",
"typography_style": "..."
}},
"copy": "...",
"image_prompt": "..."
}}
}}

每一个image_prompt必须是完整英文Prompt。

必须包含：

1. 2:3 vertical public service advertising poster
2. subject
3. environment
4. composition
5. camera / photography
6. lighting
7. emotional atmosphere
8. visual metaphor
9. typography design
10. exact Chinese slogan
11. slogan placement
12. slogan integration with artwork
13. MFT-related artistic treatment
14. high-end advertising quality

最重要：

image_prompt中必须明确写出完整中文标语。

Exact Chinese slogan to render:
“完整中文标语”

Render this exact Chinese sentence clearly and legibly.

Do not alter, translate, abbreviate, or add words.

No logo.
No watermark.
No extra text.
No unrelated typography.
""".strip()


# ============================================================
# LLM
# ============================================================

def call_llm(prompt: str):

    client = get_client(
        "LLM_API_KEY",
        "LLM_BASE_URL"
    )

    model = os.getenv(
        "LLM_MODEL",
        "gpt-5.5"
    )

    try:

        response = client.chat.completions.create(

            model=model,

            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是严谨的公益广告实验创意策略AI。"
                        "必须严格遵守MFT实验控制变量。"
                        "必须让匹配和不匹配的视觉概念具有明显区别。"
                        "必须输出合法JSON。"
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            response_format={
                "type": "json_object"
            },

            max_tokens=8000
        )

    except Exception as exc:

        raise HTTPException(
            status_code=502,
            detail=f"LLM调用失败：{exc}"
        ) from exc

    content = (
        response
        .choices[0]
        .message
        .content
        .strip()
    )

    try:

        return json.loads(content)

    except json.JSONDecodeError as exc:

        raise HTTPException(

            status_code=500,

            detail=(
                "LLM返回内容不是有效JSON。"
                f"实际返回长度：{len(content)}。"
                f"返回开头：{content[:300]}"
            )

        ) from exc


# ============================================================
# Vercel Blob
#
# 不再使用 BlobClient()
# 直接通过 Blob REST API 上传
# ============================================================

def upload_image_to_blob(
    image_bytes: bytes,
    experiment_id: str,
    condition: str
):
    """
    使用 Vercel Blob REST API 上传。

    这比 BlobClient 更适合当前 Vercel
    Python Serverless 环境。

    返回公开 Blob URL。
    """

    if not image_bytes:

        raise RuntimeError(
            "图片数据为空，无法上传 Vercel Blob。"
        )

    token = os.getenv(
        "BLOB_READ_WRITE_TOKEN"
    )

    if not token:

        raise RuntimeError(
            "没有找到 BLOB_READ_WRITE_TOKEN。"
            "请检查 Vercel Production Environment Variables。"
        )

    filename = (
        f"generated/{experiment_id}/{condition}.png"
    )

    # Vercel Blob multipart PUT API
    upload_url = (
        "https://blob.vercel-storage.com/"
        + filename
    )

    print(
        f"☁️ 开始上传 Vercel Blob：{filename}"
    )

    request = Request(
        upload_url,
        data=image_bytes,
        method="PUT",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "image/png",
            "x-api-version": "7"
        }
    )

    try:

        with urlopen(
            request,
            timeout=60
        ) as response:

            status = response.status

            response_body = response.read()

        print(
            f"☁️ Blob HTTP状态码：{status}"
        )

        if not response_body:

            raise RuntimeError(
                "Vercel Blob 返回空响应。"
            )

        try:

            blob_data = json.loads(
                response_body.decode(
                    "utf-8"
                )
            )

        except Exception:

            raise RuntimeError(
                "Vercel Blob 返回内容不是JSON："
                + response_body[:500].decode(
                    "utf-8",
                    errors="replace"
                )
            )

        blob_url = (
            blob_data.get("url")
        )

        if not blob_url:

            raise RuntimeError(
                "Vercel Blob 上传成功但没有返回 url。"
                f"返回：{blob_data}"
            )

        print(
            f"✅ Vercel Blob 上传成功：{blob_url}"
        )

        return blob_url

    except HTTPError as exc:

        error_body = ""

        try:

            error_body = exc.read().decode(
                "utf-8",
                errors="replace"
            )

        except Exception:
            pass

        print(
            f"❌ Vercel Blob HTTP错误："
            f"{exc.code} {exc.reason}"
        )

        print(
            f"❌ Blob错误响应：{error_body[:1000]}"
        )

        raise RuntimeError(
            "Vercel Blob HTTP上传失败："
            f"HTTP {exc.code} {exc.reason}; "
            f"response={error_body[:1000]}"
        ) from exc

    except URLError as exc:

        raise RuntimeError(
            f"Vercel Blob网络请求失败：{exc}"
        ) from exc

    except Exception as exc:

        raise RuntimeError(
            f"Vercel Blob上传异常："
            f"{type(exc).__name__}: {exc}"
        ) from exc


# ============================================================
# URL 图片下载
# ============================================================

def download_image_url(url: str):

    try:

        request = Request(

            url,

            headers={
                "User-Agent": (
                    "Mozilla/5.0 "
                    "(Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 "
                    "(KHTML, like Gecko) "
                    "Chrome/151.0.0.0 Safari/537.36"
                )
            }
        )

        with urlopen(
            request,
            timeout=60
        ) as response:

            data = response.read()

        if not data:

            raise RuntimeError(
                "下载到的图片为空。"
            )

        return data

    except Exception as exc:

        raise RuntimeError(
            f"图片URL下载失败：{exc}"
        ) from exc


# ============================================================
# 图片生成
# ============================================================

def generate_image(
    prompt: str,
    experiment_id: str,
    condition: str
):

    client = get_client(
        "IMAGE_API_KEY",
        "IMAGE_BASE_URL"
    )

    model = os.getenv(
        "IMAGE_MODEL",
        "gpt-image-2"
    )

    print(
        f"🎨 正在生成 {condition} 图片..."
    )

    try:

        result = client.images.generate(

            model=model,

            prompt=prompt,

            size="1024x1536",

            n=1
        )

    except Exception as exc:

        raise HTTPException(
            status_code=502,
            detail=f"图片API调用失败：{exc}"
        ) from exc

    if not result.data:

        raise HTTPException(
            status_code=502,
            detail="图片API没有返回图片数据。"
        )

    item = result.data[0]

    b64 = getattr(
        item,
        "b64_json",
        None
    )

    url = getattr(
        item,
        "url",
        None
    )

    image_bytes = None

    # ========================================================
    # b64
    # ========================================================

    if b64:

        try:

            image_bytes = base64.b64decode(
                b64
            )

            print(
                f"✅ {condition} 图片获得 b64_json"
            )

            print(
                f"📦 图片大小："
                f"{len(image_bytes) / 1024 / 1024:.2f} MB"
            )

        except Exception as exc:

            print(
                f"⚠️ b64_json解析失败：{exc}"
            )

    # ========================================================
    # URL
    # ========================================================

    if image_bytes is None and url:

        try:

            image_bytes = download_image_url(
                url
            )

            print(
                f"✅ {condition} 图片URL下载成功"
            )

        except Exception as exc:

            raise HTTPException(

                status_code=502,

                detail=(
                    f"{condition} 图片下载失败：{exc}"
                )

            ) from exc

    if image_bytes is None:

        raise HTTPException(

            status_code=502,

            detail=(
                "无法识别图片API返回格式。"
                "既没有b64_json，也没有url。"
            )
        )

    # ========================================================
    # Blob
    # ========================================================

    try:

        blob_url = upload_image_to_blob(

            image_bytes,

            experiment_id,

            condition
        )

    except Exception as exc:

        print(
            f"❌ {condition} Blob上传失败："
            f"{type(exc).__name__}: {exc}"
        )

        raise HTTPException(

            status_code=502,

            detail=(
                f"{condition} 图片上传Vercel Blob失败："
                f"{exc}"
            )

        ) from exc

    print(
        f"✅ {condition} 图片已经上传到 Vercel Blob"
    )

    return {
        "url": blob_url,
        "source": "vercel_blob"
    }


# ============================================================
# 首页
# ============================================================

@app.get("/")
def index():

    index_path = (
        BASE_DIR /
        "frontend" /
        "index.html"
    )

    if not index_path.exists():

        raise HTTPException(

            status_code=500,

            detail=(
                "找不到 frontend/index.html。"
                "请检查项目目录结构。"
            )
        )

    return FileResponse(
        index_path
    )


# ============================================================
# 健康检查
# ============================================================

@app.get("/api/health")
def health():

    database_ok = False
    database_error = None

    try:

        conn = db()

        conn.execute(
            "SELECT 1"
        )

        conn.close()

        database_ok = True

    except Exception as exc:

        database_error = str(exc)

    blob_token = os.getenv(
        "BLOB_READ_WRITE_TOKEN"
    )

    blob_store = os.getenv(
        "BLOB_STORE_ID"
    )

    return {

        "status": "ok",

        "database": database_ok,

        "database_error": database_error,

        "llm_configured": bool(
            os.getenv("LLM_API_KEY")
        ),

        "image_configured": bool(
            os.getenv("IMAGE_API_KEY")
        ),

        "blob_configured": bool(
            blob_token
        ),

        "blob_store_configured": bool(
            blob_store
        )
    }


# ============================================================
# 初始化数据库
# ============================================================

@app.post("/api/init-db")
def initialize_database():

    try:

        init_db()

        return {
            "ok": True,
            "message": "数据库初始化成功"
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"数据库初始化失败：{exc}"
        ) from exc


# ============================================================
# 生成实验
# ============================================================

@app.post("/api/generate")
def generate(
    req: GenerateRequest
):

    print(
        "\n" +
        "=" * 60
    )

    print(
        "🚀 开始生成AIGC公益广告实验"
    )

    print(
        "=" * 60
    )

    print(
        f"📌 主题：{req.topic}"
    )

    matched, unmatched = choose_conditions(
        req.mft
    )

    print(
        f"📌 匹配条件：{matched}"
    )

    print(
        f"📌 不匹配条件：{unmatched}"
    )

    # ========================================================
    # 初始化数据库
    # ========================================================

    try:

        init_db()

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"数据库初始化失败：{exc}"
        ) from exc

    # ========================================================
    # LLM
    # ========================================================

    ai = call_llm(

        build_generation_prompt(

            req.topic,

            req.mft,

            matched,

            unmatched
        )
    )

    experiment_id = (
        uuid.uuid4()
        .hex[:12]
    )

    now = datetime.now(
        timezone.utc
    )

    conn = None

    try:

        conn = db()

        matched_item = ai.get(
            "matched"
        )

        unmatched_item = ai.get(
            "unmatched"
        )

        if not matched_item:

            raise HTTPException(
                status_code=500,
                detail="LLM缺少 matched 输出。"
            )

        if not unmatched_item:

            raise HTTPException(
                status_code=500,
                detail="LLM缺少 unmatched 输出。"
            )

        if not matched_item.get(
            "image_prompt"
        ):

            raise HTTPException(
                status_code=500,
                detail="matched 缺少 image_prompt。"
            )

        if not unmatched_item.get(
            "image_prompt"
        ):

            raise HTTPException(
                status_code=500,
                detail="unmatched 缺少 image_prompt。"
            )

        # ====================================================
        # 创建实验
        # ====================================================

        conn.execute(

            """
            INSERT INTO experiments
            (
                id,
                topic,
                care,
                fairness,
                loyalty,
                authority,
                sanctity,
                matched_foundation,
                unmatched_foundation,
                created_at
            )
            VALUES
            (
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s
            )
            """,

            (
                experiment_id,
                req.topic,
                req.mft.care,
                req.mft.fairness,
                req.mft.loyalty,
                req.mft.authority,
                req.mft.sanctity,
                matched,
                unmatched,
                now
            )
        )

        # ====================================================
        # 并行生图
        # ====================================================

        image_tasks = {
            "matched": matched_item,
            "unmatched": unmatched_item
        }

        image_results = {}

        print(
            "\n🖼️ 开始并行生成两张图片..."
        )

        with ThreadPoolExecutor(
            max_workers=2
        ) as executor:

            futures = {

                executor.submit(

                    generate_image,

                    item["image_prompt"],

                    experiment_id,

                    condition

                ):
                    condition

                for condition, item
                in image_tasks.items()
            }

            for future in as_completed(
                futures
            ):

                condition = futures[
                    future
                ]

                try:

                    image_results[
                        condition
                    ] = future.result()

                    print(
                        f"✅ {condition} 图片生成完成"
                    )

                except Exception as exc:

                    print(
                        f"❌ {condition} 图片生成失败："
                        f"{type(exc).__name__}: {exc}"
                    )

                    raise

        # ====================================================
        # 保存广告
        # ====================================================

        result = {

            "experiment_id":
                experiment_id,

            "topic":
                req.topic,

            "mft":
                scores_dict(req.mft),

            "matched_foundation":
                matched,

            "unmatched_foundation":
                unmatched,

            "matched":
                None,

            "unmatched":
                None
        }

        for condition, foundation, key in [

            (
                "matched",
                matched,
                "matched"
            ),

            (
                "unmatched",
                unmatched,
                "unmatched"
            )

        ]:

            item = ai[key]

            ad_id = (
                uuid.uuid4()
                .hex[:12]
            )

            image_url = (
                image_results[
                    condition
                ]["url"]
            )

            strategy = item.get(
                "strategy",
                {}
            )

            copy_text = item.get(
                "copy",
                ""
            )

            image_prompt = item.get(
                "image_prompt",
                ""
            )

            conn.execute(

                """
                INSERT INTO ads
                (
                    id,
                    experiment_id,
                    condition,
                    foundation,
                    strategy_json,
                    copy,
                    image_prompt,
                    image_url
                )
                VALUES
                (
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s
                )
                """,

                (
                    ad_id,
                    experiment_id,
                    condition,
                    foundation,
                    json.dumps(
                        strategy,
                        ensure_ascii=False
                    ),
                    copy_text,
                    image_prompt,
                    image_url
                )
            )

            result[condition] = {

                "ad_id":
                    ad_id,

                "foundation":
                    foundation,

                "strategy":
                    strategy,

                "copy":
                    copy_text,

                "image_prompt":
                    image_prompt,

                "image_url":
                    image_url
            }

        conn.commit()

        print(
            "\n" +
            "=" * 60
        )

        print(
            "🎉 整个实验生成完成"
        )

        print(
            "=" * 60 +
            "\n"
        )

        return result

    except HTTPException:

        if conn:
            conn.rollback()

        raise

    except Exception as exc:

        if conn:
            conn.rollback()

        raise HTTPException(

            status_code=502,

            detail=(
                f"实验生成失败："
                f"{type(exc).__name__}: {exc}"
            )

        ) from exc

    finally:

        if conn:
            conn.close()


# ============================================================
# 提交评价
# ============================================================

@app.post("/api/evaluate")
def evaluate(
    req: EvaluationRequest
):

    conn = None

    try:

        conn = db()

        exists = conn.execute(

            """
            SELECT id
            FROM ads
            WHERE id = %s
            AND experiment_id = %s
            """,

            (
                req.ad_id,
                req.experiment_id
            )

        ).fetchone()

        if not exists:

            raise HTTPException(

                status_code=404,

                detail="找不到对应广告。"
            )

        conn.execute(

            """
            INSERT INTO evaluations
            (
                experiment_id,
                ad_id,
                moral_resonance,
                emotional_response,
                persuasion,
                behavioral_intention,
                created_at
            )
            VALUES
            (
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s
            )
            """,

            (
                req.experiment_id,
                req.ad_id,
                req.moral_resonance,
                req.emotional_response,
                req.persuasion,
                req.behavioral_intention,
                datetime.now(timezone.utc)
            )
        )

        conn.commit()

        return {
            "ok": True
        }

    except HTTPException:

        if conn:
            conn.rollback()

        raise

    except Exception as exc:

        if conn:
            conn.rollback()

        raise HTTPException(

            status_code=500,

            detail=(
                f"评价保存失败：{exc}"
            )

        ) from exc

    finally:

        if conn:
            conn.close()


# ============================================================
# 历史记录
# ============================================================

@app.get("/api/history")
def get_history():

    conn = None

    try:

        init_db()

        conn = db()

        rows = conn.execute(

            """
            SELECT
                e.id,
                e.topic,
                e.care,
                e.fairness,
                e.loyalty,
                e.authority,
                e.sanctity,
                e.matched_foundation,
                e.unmatched_foundation,
                e.created_at,
                COUNT(DISTINCT a.id) AS ad_count,
                COUNT(DISTINCT ev.id) AS evaluation_count

            FROM experiments e

            LEFT JOIN ads a
                ON e.id = a.experiment_id

            LEFT JOIN evaluations ev
                ON e.id = ev.experiment_id

            GROUP BY
                e.id,
                e.topic,
                e.care,
                e.fairness,
                e.loyalty,
                e.authority,
                e.sanctity,
                e.matched_foundation,
                e.unmatched_foundation,
                e.created_at

            ORDER BY e.created_at DESC
            """
        ).fetchall()

        history = []

        for row in rows:

            history.append({

                "experiment_id":
                    row["id"],

                "topic":
                    row["topic"],

                "mft": {

                    "care":
                        row["care"],

                    "fairness":
                        row["fairness"],

                    "loyalty":
                        row["loyalty"],

                    "authority":
                        row["authority"],

                    "sanctity":
                        row["sanctity"]
                },

                "matched_foundation":
                    row["matched_foundation"],

                "unmatched_foundation":
                    row["unmatched_foundation"],

                "created_at":
                    row["created_at"].isoformat()
                    if row["created_at"]
                    else None,

                "ad_count":
                    row["ad_count"],

                "evaluation_count":
                    row["evaluation_count"]
            })

        return {

            "total":
                len(history),

            "history":
                history
        }

    except Exception as exc:

        raise HTTPException(

            status_code=500,

            detail=(
                f"历史记录读取失败：{exc}"
            )

        ) from exc

    finally:

        if conn:
            conn.close()


# ============================================================
# 历史实验详情
# ============================================================

@app.get(
    "/api/history/{experiment_id}"
)
def get_history_detail(
    experiment_id: str
):

    conn = None

    try:

        conn = db()

        experiment = conn.execute(

            """
            SELECT *
            FROM experiments
            WHERE id = %s
            """,

            (
                experiment_id,
            )

        ).fetchone()

        if not experiment:

            raise HTTPException(

                status_code=404,

                detail="找不到该历史实验。"
            )

        ads = conn.execute(

            """
            SELECT *
            FROM ads
            WHERE experiment_id = %s
            ORDER BY
                CASE
                    WHEN condition = 'matched'
                    THEN 1
                    ELSE 2
                END
            """,

            (
                experiment_id,
            )

        ).fetchall()

        evaluations = conn.execute(

            """
            SELECT *
            FROM evaluations
            WHERE experiment_id = %s
            ORDER BY created_at ASC
            """,

            (
                experiment_id,
            )

        ).fetchall()

        result = {

            "experiment_id":
                experiment["id"],

            "topic":
                experiment["topic"],

            "mft": {

                "care":
                    experiment["care"],

                "fairness":
                    experiment["fairness"],

                "loyalty":
                    experiment["loyalty"],

                "authority":
                    experiment["authority"],

                "sanctity":
                    experiment["sanctity"]
            },

            "matched_foundation":
                experiment[
                    "matched_foundation"
                ],

            "unmatched_foundation":
                experiment[
                    "unmatched_foundation"
                ],

            "created_at":
                experiment["created_at"].isoformat()
                if experiment["created_at"]
                else None,

            "matched":
                None,

            "unmatched":
                None,

            "evaluations":
                []
        }

        for ad in ads:

            try:

                strategy = json.loads(
                    ad["strategy_json"]
                )

            except Exception:

                strategy = {}

            item = {

                "ad_id":
                    ad["id"],

                "foundation":
                    ad["foundation"],

                "strategy":
                    strategy,

                "copy":
                    ad["copy"],

                "image_prompt":
                    ad["image_prompt"],

                "image_url":
                    ad["image_url"]
            }

            if ad["condition"] == "matched":

                result["matched"] = item

            else:

                result["unmatched"] = item

        for evaluation in evaluations:

            result["evaluations"].append({

                "id":
                    evaluation["id"],

                "ad_id":
                    evaluation["ad_id"],

                "moral_resonance":
                    evaluation["moral_resonance"],

                "emotional_response":
                    evaluation["emotional_response"],

                "persuasion":
                    evaluation["persuasion"],

                "behavioral_intention":
                    evaluation["behavioral_intention"],

                "created_at":
                    evaluation["created_at"].isoformat()
                    if evaluation["created_at"]
                    else None
            })

        return result

    except HTTPException:

        raise

    except Exception as exc:

        raise HTTPException(

            status_code=500,

            detail=(
                f"历史实验详情读取失败：{exc}"
            )

        ) from exc

    finally:

        if conn:
            conn.close()
