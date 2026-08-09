from pathlib import Path
import base64
import json
import os
import time
import uuid
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from openai import OpenAI
from pydantic import BaseModel, Field

import psycopg
from psycopg.rows import dict_row

from vercel.blob import BlobClient


# ============================================================
# 基础配置
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# 本地开发时读取 .env
# Vercel 部署时直接使用 Environment Variables
load_dotenv(BASE_DIR / ".env")

app = FastAPI(
    title="AIGC MFT 公益广告实验平台",
    version="2.0.0"
)


# ============================================================
# Vercel Serverless 配置
# ============================================================

# 防止某些部署环境将当前工作目录设置得不确定
os.environ.setdefault(
    "PYTHONUNBUFFERED",
    "1"
)


# ============================================================
# PostgreSQL
# ============================================================

DATABASE_URL = os.getenv("DATABASE_URL")


def db():
    """
    创建 PostgreSQL 连接。

    Vercel:
        使用 DATABASE_URL。

    本地:
        使用 .env 中的 DATABASE_URL。
    """

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError(
            "未配置 DATABASE_URL。"
            "请在 Vercel Project Settings → Environment Variables "
            "中配置 DATABASE_URL。"
        )

    return psycopg.connect(
        database_url,
        row_factory=dict_row,
        connect_timeout=10
    )


def init_db():
    """
    初始化 PostgreSQL 数据表。

    注意：
    Vercel Serverless 中不能依赖本地 SQLite。
    所有持久化数据都放 PostgreSQL。
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
# 不在 import 阶段让整个 Serverless Function 崩溃
# ============================================================

try:

    init_db()

    print("✅ PostgreSQL 数据库初始化检查完成")

except Exception as exc:

    print(
        f"⚠️ 数据库初始化检查失败：{exc}"
    )

    print(
        "⚠️ 不阻止 FastAPI 启动，API 请求时会返回具体数据库错误。"
    )


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
# Pydantic 数据模型
# ============================================================

class MFTScores(BaseModel):

    care: float = Field(
        ge=0,
        le=10
    )

    fairness: float = Field(
        ge=0,
        le=10
    )

    loyalty: float = Field(
        ge=0,
        le=10
    )

    authority: float = Field(
        ge=0,
        le=10
    )

    sanctity: float = Field(
        ge=0,
        le=10
    )


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
    """
    创建 OpenAI-compatible Client。

    适用于：
    - 官方 OpenAI API
    - 你的中转站
    - 其他 OpenAI Compatible API
    """

    key = os.getenv(api_key_name)

    base_url = os.getenv(base_url_name)

    if not key or key.startswith("your_"):

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

    return OpenAI(
        **kwargs,
        timeout=180.0,
        max_retries=2
    )


# ============================================================
# MFT
# ============================================================

def scores_dict(
    mft: MFTScores
):

    return {

        "care":
            mft.care,

        "fairness":
            mft.fairness,

        "loyalty":
            mft.loyalty,

        "authority":
            mft.authority,

        "sanctity":
            mft.sanctity
    }


def choose_conditions(
    mft: MFTScores
):

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

必须根据该MFT维度选择：

- 核心视觉主体
- 视觉隐喻
- 场景
- 色彩倾向
- 情绪
- 摄影方式
- 构图
- 字体风格
- 标语排版方式

MFT参考：

Care / Harm：

强调生命、保护、伤害、脆弱、陪伴、救助。

可以使用动物、儿童、自然生命、受伤的生命等。

Fairness / Cheating：

强调公平、不公平、交换、失衡、规则、机会差距。

可以使用天平、两种待遇、资源分配、对比场景等。

Loyalty / Betrayal：

强调共同体、承诺、责任、背叛、关系。

可以使用人与人、群体、家庭、伙伴、共同守护等。

Authority / Subversion：

强调规则、秩序、责任、公共规范、社会制度。

可以使用城市、道路、公共设施、规则标识、秩序结构等。

Sanctity / Degradation：

强调纯净、污染、神圣、洁净与肮脏之间的冲突。

可以使用纯净自然、污染侵入、洁净空间被破坏等。

不匹配版必须围绕：

{FOUNDATIONS[unmatched]['cn']}

展开。

但是视觉概念必须与匹配版明显不同。

禁止：

“匹配版是一只鲸鱼，
不匹配版还是一只鲸鱼，只是换成了规则概念。”

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

例如：

如果匹配版使用“动物受到伤害”，
不匹配版可以使用“城市公共规则”。

如果匹配版使用“家庭关系”，
不匹配版可以使用“公共空间”。

如果匹配版使用“自然生命”，
不匹配版可以使用“人工设施”。

这样才能形成明显的实验刺激差异。

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

例如：

匹配版：

“少一份伤害，多一条生命的路。”

不匹配版：

“守一份规则，多一片海洋的净土。”

两个版本的句式具有对应关系，
但道德诉求明显不同。

这是一个海报生成任务。

图片中必须直接出现对应的中文公益广告标语。

禁止把标语留给网页后期叠加。

image_prompt必须明确写出：

EXACT SLOGAN TO RENDER:

“这里放copy中的完整中文标语”

图像模型必须尝试将这句话：

逐字、完整、清晰地生成在海报中。

不得修改文字。

不得增加额外文字。

不得删除文字。

不得生成英文翻译。

不得生成乱码。

不得生成假文字。

不得生成Logo。

不得生成水印。

标语不仅仅是普通文字。

必须根据MFT道德基础设计文字美术效果。

Care / Harm：

可以使用柔和、温暖、具有保护感的字体。

文字可以与光线、生命、手掌、自然元素融合。

Fairness / Cheating：

可以使用对称排版、平衡构图、类似天平的视觉结构。

文字可以具有强烈的秩序感和几何感。

Loyalty / Betrayal：

可以使用具有连接感、手写感、纪念感或共同体感的字体。

文字可以与人物关系或连接线形成视觉呼应。

Authority / Subversion：

可以使用强烈、规整、现代、公共标识风格的字体。

文字可以采用类似公共警示牌、城市标识、规则系统的排版。

Sanctity / Degradation：

可以使用高对比、洁净、庄重的字体。

文字可以与纯净空间、光束、污染边界形成视觉冲突。

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

结构必须严格为：

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

例如：

Exact Chinese slogan to render:
“少一份伤害，多一条生命的路。”

Render this exact Chinese sentence clearly and legibly.

Do not alter, translate, abbreviate, or add words.

最后：

No logo.
No watermark.
No extra text.
No unrelated typography.
""".strip()


# ============================================================
# LLM 调用
# ============================================================

def call_llm(
    prompt: str
):

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

    if not response.choices:

        raise HTTPException(
            status_code=502,
            detail="LLM没有返回任何choices。"
        )

    content = (
        response
        .choices[0]
        .message
        .content
    )

    if not content:

        raise HTTPException(
            status_code=502,
            detail="LLM返回内容为空。"
        )

    content = content.strip()

    try:

        return json.loads(
            content
        )

    except json.JSONDecodeError as exc:

        raise HTTPException(

            status_code=500,

            detail=(
                "LLM返回内容不是有效JSON。"
                f"实际返回长度：{len(content)} 字符。"
                f"返回开头：{content[:500]}"
            )

        ) from exc


# ============================================================
# Vercel Blob
# ============================================================

def get_blob_token():

    """
    获取 Vercel Blob Read/Write Token。

    优先：
        BLOB_READ_WRITE_TOKEN

    兼容：
        VERCEL_BLOB_READ_WRITE_TOKEN

    注意：
    BLOB_STORE_ID 不能替代 READ_WRITE_TOKEN。
    """

    token = os.getenv(
        "BLOB_READ_WRITE_TOKEN"
    )

    if not token:

        token = os.getenv(
            "VERCEL_BLOB_READ_WRITE_TOKEN"
        )

    return token


def upload_image_to_blob(
    image_bytes: bytes,
    experiment_id: str,
    condition: str
):
    """
    上传图片到 Vercel Blob。

    Vercel Serverless 中：
    - 不保存到本地
    - 不使用 /tmp 作为永久存储
    - 直接将 bytes 上传到 Blob
    """

    if not image_bytes:

        raise RuntimeError(
            "图片数据为空，无法上传到 Vercel Blob。"
        )

    token = get_blob_token()

    if not token:

        raise RuntimeError(
            "未找到 BLOB_READ_WRITE_TOKEN。"
            "请在 Vercel Environment Variables 中配置。"
        )

    store_id = os.getenv(
        "BLOB_STORE_ID"
    )

    if not store_id:

        print(
            "⚠️ 未检测到 BLOB_STORE_ID，"
            "但继续尝试使用 BLOB_READ_WRITE_TOKEN 上传。"
        )

    filename = (
        f"generated/{experiment_id}/{condition}.png"
    )

    print(
        f"☁️ 开始上传 Blob：{filename}"
    )

    print(
        f"☁️ 图片大小：{len(image_bytes) / 1024 / 1024:.2f} MB"
    )

    # ========================================================
    # Vercel Blob Python SDK
    # ========================================================

    last_error = None

    for attempt in range(1, 4):

        try:

            client = BlobClient(
                token=token
            )

            blob = client.put(

                filename,

                image_bytes,

                access="public",

                content_type="image/png",

                add_random_suffix=False,

                overwrite=True

            )

            blob_url = getattr(
                blob,
                "url",
                None
            )

            if not blob_url:

                raise RuntimeError(
                    "Vercel Blob 上传成功但没有返回 URL。"
                )

            print(
                f"✅ Blob 上传成功：{blob_url}"
            )

            return blob_url

        except Exception as exc:

            last_error = exc

            print(
                f"⚠️ Blob 上传第 {attempt}/3 次失败："
                f"{type(exc).__name__}: {exc}"
            )

            if attempt < 3:

                time.sleep(
                    attempt * 2
                )

    raise RuntimeError(
        "Vercel Blob 上传失败，"
        f"已重试3次。"
        f"最后错误："
        f"{type(last_error).__name__}: {last_error}"
    )


# ============================================================
# URL图片下载
# ============================================================

def download_image_url(
    url: str
):

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
            timeout=90
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
            detail=f"{condition} 图片API调用失败：{exc}"
        ) from exc

    if not result.data:

        raise HTTPException(
            status_code=502,
            detail=f"{condition} 图片API没有返回图片数据。"
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
    # 方案1：b64_json
    # ========================================================

    if b64:

        try:

            image_bytes = base64.b64decode(
                b64,
                validate=True
            )

            print(
                f"✅ {condition} 图片获得 b64_json"
            )

        except Exception as exc:

            print(
                f"⚠️ {condition} b64_json解析失败：{exc}"
            )

    # ========================================================
    # 方案2：URL
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
                f"{condition} 图片API返回格式无法识别。"
                "既没有b64_json，也没有url。"
            )
        )

    # ========================================================
    # 上传 Vercel Blob
    # ========================================================

    try:

        blob_url = upload_image_to_blob(

            image_bytes,

            experiment_id,

            condition
        )

    except Exception as exc:

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

        "url":
            blob_url,

        "source":
            "vercel_blob"
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

    blob_token = get_blob_token()

    return {

        "status":
            "ok",

        "database":
            database_ok,

        "database_error":
            database_error,

        "llm_configured":
            bool(
                os.getenv(
                    "LLM_API_KEY"
                )
            ),

        "llm_base_url_configured":
            bool(
                os.getenv(
                    "LLM_BASE_URL"
                )
            ),

        "image_configured":
            bool(
                os.getenv(
                    "IMAGE_API_KEY"
                )
            ),

        "image_base_url_configured":
            bool(
                os.getenv(
                    "IMAGE_BASE_URL"
                )
            ),

        "blob_configured":
            bool(blob_token),

        "blob_store_configured":
            bool(
                os.getenv(
                    "BLOB_STORE_ID"
                )
            )
    }


# ============================================================
# Blob 专用诊断
# ============================================================

@app.get("/api/blob-health")
def blob_health():

    token = get_blob_token()

    store_id = os.getenv(
        "BLOB_STORE_ID"
    )

    if not token:

        return {

            "ok":
                False,

            "message":
                "BLOB_READ_WRITE_TOKEN 未配置。",

            "token_configured":
                False,

            "store_configured":
                bool(store_id)
        }

    try:

        client = BlobClient(
            token=token
        )

        # 不上传测试文件。
        # 这里仅确认 SDK 可以创建客户端。
        return {

            "ok":
                True,

            "message":
                "Vercel Blob Client 创建成功。",

            "token_configured":
                True,

            "store_configured":
                bool(store_id),

            "store_id":
                store_id
        }

    except Exception as exc:

        return {

            "ok":
                False,

            "message":
                f"Vercel Blob Client 初始化失败：{exc}",

            "token_configured":
                True,

            "store_configured":
                bool(store_id)
        }


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
        f"📌 公益主题：{req.topic}"
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
    # 第一步：LLM生成策略
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

    # ========================================================
    # 检查LLM输出
    # ========================================================

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

    # ========================================================
    # 第二步：数据库创建实验记录
    # ========================================================

    conn = None

    try:

        conn = db()

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

        conn.commit()

    except Exception as exc:

        if conn:

            conn.rollback()

            conn.close()

        raise HTTPException(

            status_code=500,

            detail=(
                f"实验数据库记录创建失败：{exc}"
            )

        ) from exc

    finally:

        if conn:

            conn.close()

    # ========================================================
    # 第三步：并行生成两张图片
    # ========================================================

    image_tasks = {

        "matched":
            matched_item,

        "unmatched":
            unmatched_item
    }

    image_results = {}

    print(
        "\n🖼️ 开始并行生成两张图片..."
    )

    try:

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

    except HTTPException:

        raise

    except Exception as exc:

        raise HTTPException(

            status_code=502,

            detail=(
                f"图片生成阶段失败：{exc}"
            )

        ) from exc

    # ========================================================
    # 第四步：写入广告数据
    # ========================================================

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

    conn = None

    try:

        conn = db()

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

    except Exception as exc:

        if conn:

            conn.rollback()

        raise HTTPException(

            status_code=500,

            detail=(
                f"广告数据保存失败：{exc}"
            )

        ) from exc

    finally:

        if conn:

            conn.close()

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

                datetime.now(
                    timezone.utc
                )
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
# 历史实验记录列表
# ============================================================

@app.get("/api/history")
def get_history():

    conn = None

    try:

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

                COUNT(DISTINCT a.id)
                    AS ad_count,

                COUNT(DISTINCT ev.id)
                    AS evaluation_count

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

            ORDER BY
                e.created_at DESC
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
                    row[
                        "matched_foundation"
                    ],

                "unmatched_foundation":
                    row[
                        "unmatched_foundation"
                    ],

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
# 获取单个历史实验详情
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

            ORDER BY
                created_at ASC
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

                result[
                    "matched"
                ] = item

            else:

                result[
                    "unmatched"
                ] = item

        for evaluation in evaluations:

            result[
                "evaluations"
            ].append({

                "id":
                    evaluation["id"],

                "ad_id":
                    evaluation["ad_id"],

                "moral_resonance":
                    evaluation[
                        "moral_resonance"
                    ],

                "emotional_response":
                    evaluation[
                        "emotional_response"
                    ],

                "persuasion":
                    evaluation[
                        "persuasion"
                    ],

                "behavioral_intention":
                    evaluation[
                        "behavioral_intention"
                    ],

                "created_at":
                    evaluation[
                        "created_at"
                    ].isoformat()
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


# ============================================================
# Vercel / ASGI
# ============================================================

# 不需要 uvicorn.run()
#
# Vercel 会自动发现：
#
#     app = FastAPI(...)
#
# 并通过 ASGI 运行。
#
# 本地运行时使用：
#
#     uvicorn backend.main:app --reload
#
# 或：
#
#     python -m uvicorn backend.main:app --reload
#
# ============================================================


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(
            os.getenv(
                "PORT",
                "8000"
            )
        )
    )
