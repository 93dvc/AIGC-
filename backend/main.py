from pathlib import Path
import base64
import json
import os
import uuid
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from openai import OpenAI
from pydantic import BaseModel, Field
import psycopg
from psycopg.rows import dict_row


# ============================================================
# 基础配置
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

app = FastAPI(
    title="AIGC 个性化公益广告实验平台"
)


# ============================================================
# 环境变量
# ============================================================

DATABASE_URL = os.getenv("DATABASE_URL")

LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5.5")

IMAGE_API_KEY = os.getenv("IMAGE_API_KEY")
IMAGE_BASE_URL = os.getenv("IMAGE_BASE_URL")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gpt-image-2")

BLOB_READ_WRITE_TOKEN = os.getenv(
    "BLOB_READ_WRITE_TOKEN"
)

BLOB_STORE_ID = os.getenv(
    "BLOB_STORE_ID"
)


# ============================================================
# 数据库
# ============================================================

def db():
    if not DATABASE_URL:
        raise RuntimeError(
            "未配置 DATABASE_URL。"
        )

    return psycopg.connect(
        DATABASE_URL,
        row_factory=dict_row,
        connect_timeout=15
    )


def init_db():
    """
    自动创建 / 迁移数据库。

    特别处理旧版 ads 表：
    如果以前没有 created_at，
    会自动 ALTER TABLE ADD COLUMN。
    """

    conn = db()

    try:
        # ----------------------------------------------------
        # experiments
        # ----------------------------------------------------

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                topic TEXT NOT NULL,
                care DOUBLE PRECISION NOT NULL DEFAULT 0,
                fairness DOUBLE PRECISION NOT NULL DEFAULT 0,
                loyalty DOUBLE PRECISION NOT NULL DEFAULT 0,
                authority DOUBLE PRECISION NOT NULL DEFAULT 0,
                sanctity DOUBLE PRECISION NOT NULL DEFAULT 0,
                matched_foundation TEXT NOT NULL,
                unmatched_foundation TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # ----------------------------------------------------
        # experiments 旧字段迁移
        # ----------------------------------------------------

        experiment_columns = [
            (
                "care",
                "DOUBLE PRECISION NOT NULL DEFAULT 0"
            ),
            (
                "fairness",
                "DOUBLE PRECISION NOT NULL DEFAULT 0"
            ),
            (
                "loyalty",
                "DOUBLE PRECISION NOT NULL DEFAULT 0"
            ),
            (
                "authority",
                "DOUBLE PRECISION NOT NULL DEFAULT 0"
            ),
            (
                "sanctity",
                "DOUBLE PRECISION NOT NULL DEFAULT 0"
            ),
            (
                "matched_foundation",
                "TEXT NOT NULL DEFAULT 'care'"
            ),
            (
                "unmatched_foundation",
                "TEXT NOT NULL DEFAULT 'authority'"
            ),
            (
                "created_at",
                "TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP"
            ),
        ]

        for column, definition in experiment_columns:
            conn.execute(
                f"""
                ALTER TABLE experiments
                ADD COLUMN IF NOT EXISTS {column}
                {definition}
                """
            )

        # ----------------------------------------------------
        # ads
        # ----------------------------------------------------

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ads (
                id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                condition TEXT NOT NULL,
                foundation TEXT NOT NULL,
                strategy_json TEXT NOT NULL DEFAULT '{}',
                copy TEXT NOT NULL DEFAULT '',
                image_prompt TEXT NOT NULL DEFAULT '',
                image_url TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(experiment_id)
                    REFERENCES experiments(id)
                    ON DELETE CASCADE
            )
            """
        )

        # ----------------------------------------------------
        # ads 旧字段迁移
        # ----------------------------------------------------

        ad_columns = [
            (
                "strategy_json",
                "TEXT NOT NULL DEFAULT '{}'"
            ),
            (
                "copy",
                "TEXT NOT NULL DEFAULT ''"
            ),
            (
                "image_prompt",
                "TEXT NOT NULL DEFAULT ''"
            ),
            (
                "image_url",
                "TEXT"
            ),
            (
                "created_at",
                "TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP"
            ),
        ]

        for column, definition in ad_columns:
            conn.execute(
                f"""
                ALTER TABLE ads
                ADD COLUMN IF NOT EXISTS {column}
                {definition}
                """
            )

        # ----------------------------------------------------
        # evaluations
        # ----------------------------------------------------

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
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(experiment_id)
                    REFERENCES experiments(id)
                    ON DELETE CASCADE,
                FOREIGN KEY(ad_id)
                    REFERENCES ads(id)
                    ON DELETE CASCADE
            )
            """
        )

        conn.execute(
            """
            ALTER TABLE evaluations
            ADD COLUMN IF NOT EXISTS created_at
            TIMESTAMPTZ NOT NULL
            DEFAULT CURRENT_TIMESTAMP
            """
        )

        conn.commit()

        print("✅ PostgreSQL 数据库初始化/迁移成功")

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


def ensure_db():
    """
    Serverless 每次需要数据库时确保结构存在。
    """

    try:
        init_db()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"数据库初始化失败：{exc}"
        ) from exc


# ============================================================
# MFT
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


class GenerateImageRequest(BaseModel):
    experiment_id: str
    condition: str
    image_prompt: str = Field(
        min_length=10,
        max_length=30000
    )


class FinalizeRequest(BaseModel):
    experiment_id: str
    topic: str
    mft: MFTScores

    matched_foundation: str
    unmatched_foundation: str

    matched: dict
    unmatched: dict


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
# 工具
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

这是一个严格的实验刺激生成任务。

必须保持：

相同公益主题。

但是匹配版和不匹配版必须使用明显不同的道德诉求。

禁止仅仅改变颜色、几个关键词或者同一主体。

两个版本必须明显改变：

1. 核心主体
2. 场景
3. 视觉隐喻
4. 构图
5. 情绪
6. 视觉符号

匹配版必须围绕：

{FOUNDATIONS[matched]['cn']}

展开。

不匹配版必须围绕：

{FOUNDATIONS[unmatched]['cn']}

展开。

MFT参考：

Care / Harm：
生命、保护、伤害、脆弱、陪伴、救助。

Fairness / Cheating：
公平、不公平、交换、失衡、规则、机会差距。

Loyalty / Betrayal：
共同体、承诺、责任、背叛、家庭、伙伴。

Authority / Subversion：
规则、秩序、责任、公共规范、社会制度。

Sanctity / Degradation：
纯净、污染、神圣、洁净与肮脏的冲突。

必须生成两句中文公益广告标语。

要求：

1. 两句围绕完全相同的公益主题。
2. 两句具有明显对仗关系。
3. 句式长度尽量接近。
4. 不能出现学术术语。
5. 必须像真实公益广告。
6. 必须体现不同MFT道德价值。
7. 具有传播性和记忆点。

图片中必须直接出现对应中文标语。

image_prompt必须明确包含：

Exact Chinese slogan to render:
“完整中文标语”

Render this exact Chinese sentence clearly and legibly.

Do not alter, translate, abbreviate, or add words.

禁止：

Logo
Watermark
品牌名称
网址
额外文字
随机英文
随机字母
乱码

图片要求：

2:3 vertical public service advertising poster.

1024x1536.

必须具有：

明确视觉主体
明确视觉焦点
前景、中景、背景层次
专业广告摄影或艺术设计
强烈视觉隐喻
合理文字区域
高质量商业广告完成度

标语位置不限。

必须清晰完整可读。

匹配版和不匹配版的摄影质量、分辨率、完成度必须保持同等级。

不要Markdown。

不要代码块。

只输出合法JSON。

严格结构：

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

image_prompt必须是完整英文Prompt。

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

最后：

No logo.
No watermark.
No extra text.
No unrelated typography.
""".strip()


# ============================================================
# LLM
# ============================================================

def get_llm_client():

    if not LLM_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="未配置 LLM_API_KEY。"
        )

    kwargs = {
        "api_key": LLM_API_KEY
    }

    if LLM_BASE_URL:
        kwargs["base_url"] = LLM_BASE_URL

    return OpenAI(**kwargs)


# ============================================================
# LLM 调用
# ============================================================

def call_llm(prompt: str):
    """
    调用 LLM。

    针对第三方 API / Cloudflare 的 502：
    - 不在 Vercel 中傻等 60 秒
    - 识别 retryable 502
    - 返回 503
    - 给前端明确的重试提示
    """

    print("🤖 开始调用 LLM...")
    print(f"🤖 模型：{LLM_MODEL}")
    print(f"🤖 Base URL：{LLM_BASE_URL}")

    client = get_llm_client()

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是严谨的公益广告实验创意策略AI。"
                        "必须严格遵守MFT实验控制变量。"
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

            # 原来是 8000
            # 当前任务不需要这么大的输出空间
            max_tokens=4000
        )

    except Exception as exc:

        # ----------------------------------------------------
        # 获取 HTTP 状态码
        # ----------------------------------------------------

        status_code = getattr(
            exc,
            "status_code",
            None
        )

        error_text = str(exc)

        print("=" * 60)
        print("❌ LLM 调用失败")
        print(f"❌ 类型：{type(exc).__name__}")
        print(f"❌ 状态码：{status_code}")
        print(f"❌ 错误：{error_text[:2000]}")
        print("=" * 60)

        # ----------------------------------------------------
        # Cloudflare / 第三方 API 502
        # ----------------------------------------------------

        if status_code == 502 or "502" in error_text:

            # Cloudflare 明确返回 retry_after=60
            retry_after = 60

            # 如果错误信息里包含 retry_after，尝试提取
            import re

            match = re.search(
                r"retry_after['\"]?\s*[:=]\s*(\d+)",
                error_text,
                re.IGNORECASE
            )

            if match:
                try:
                    retry_after = int(match.group(1))
                except Exception:
                    retry_after = 60

            print(
                f"⚠️ 上游 LLM 服务暂时不可用。"
                f"建议 {retry_after} 秒后重试。"
            )

            # 注意：
            # 不要在 Vercel 里 time.sleep(60)
            # 否则很容易造成 Serverless 请求超时。
            raise HTTPException(
                status_code=503,
                headers={
                    "Retry-After": str(retry_after)
                },
                detail=(
                    "AI服务暂时繁忙，"
                    f"上游服务器返回 Cloudflare 502。"
                    f"请等待约 {retry_after} 秒后重新生成。"
                )
            ) from exc

        # ----------------------------------------------------
        # 429：请求过多
        # ----------------------------------------------------

        if status_code == 429:

            print("⚠️ LLM API 请求过于频繁。")

            raise HTTPException(
                status_code=429,
                headers={
                    "Retry-After": "30"
                },
                detail=(
                    "AI服务请求过于频繁，"
                    "请稍等 30 秒后再试。"
                )
            ) from exc

        # ----------------------------------------------------
        # 其他 5xx
        # ----------------------------------------------------

        if status_code is not None and status_code >= 500:

            raise HTTPException(
                status_code=503,
                detail=(
                    "AI服务暂时不可用。"
                    f"上游返回 HTTP {status_code}，"
                    "请稍后重试。"
                )
            ) from exc

        # ----------------------------------------------------
        # API Key / 权限问题
        # ----------------------------------------------------

        if status_code in (401, 403):

            raise HTTPException(
                status_code=502,
                detail=(
                    "LLM API 鉴权失败，请检查 "
                    "LLM_API_KEY 是否正确。"
                )
            ) from exc

        # ----------------------------------------------------
        # 其他未知错误
        # ----------------------------------------------------

        raise HTTPException(
            status_code=502,
            detail=f"LLM调用失败：{error_text[:1500]}"
        ) from exc

    # ========================================================
    # 检查返回结果
    # ========================================================

    if not response.choices:
        raise HTTPException(
            status_code=502,
            detail="LLM没有返回有效结果。"
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

    print(
        f"📄 LLM返回长度：{len(content)} 字符"
    )

    # ========================================================
    # JSON解析
    # ========================================================

    try:

        data = json.loads(content)

    except json.JSONDecodeError as exc:

        print("❌ LLM返回的内容不是合法JSON")
        print(
            f"❌ 内容开头：{content[:1000]}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "LLM返回内容不是有效JSON。"
                f"长度：{len(content)}；"
                f"开头：{content[:300]}"
            )
        ) from exc

    print("✅ LLM策略生成完成")

    return data

# ============================================================
# Blob
# ============================================================

def upload_image_to_blob(
    image_bytes: bytes,
    experiment_id: str,
    condition: str
):

    if not BLOB_READ_WRITE_TOKEN:
        raise RuntimeError(
            "未配置 BLOB_READ_WRITE_TOKEN。"
        )

    if not image_bytes:
        raise RuntimeError(
            "图片数据为空。"
        )

    filename = (
        f"generated/"
        f"{experiment_id}/"
        f"{condition}.png"
    )

    print(
        f"📦 图片大小："
        f"{len(image_bytes) / 1024 / 1024:.2f} MB"
    )

    print(
        f"☁️ 开始上传 Vercel Blob："
        f"{filename}"
    )

    blob_url = (
        "https://blob.vercel-storage.com/"
        + filename
    )

    request = Request(
        blob_url,
        data=image_bytes,
        method="PUT",
        headers={
            "Authorization":
                f"Bearer {BLOB_READ_WRITE_TOKEN}",

            "x-content-type":
                "image/png",

            "Content-Type":
                "image/png",

            "x-vercel-blob-access":
                "public",

            "x-vercel-blob-add-random-suffix":
                "true"
        }
    )

    try:

        with urlopen(
            request,
            timeout=60
        ) as response:

            status = response.status

            body = response.read()

        print(
            f"☁️ Blob HTTP状态码：{status}"
        )

        if status not in (200, 201):

            raise RuntimeError(
                f"Blob返回HTTP {status}: "
                f"{body[:500]!r}"
            )

        # Blob PUT API通常返回JSON
        try:

            payload = json.loads(
                body.decode("utf-8")
            )

            returned_url = (
                payload.get("url")
                or payload.get("downloadUrl")
            )

        except Exception:

            returned_url = None

        if not returned_url:

            raise RuntimeError(
                "Blob上传成功但没有返回图片URL。"
            )

        print(
            f"✅ Vercel Blob 上传成功："
            f"{returned_url}"
        )

        return returned_url

    except HTTPError as exc:

        detail = exc.read().decode(
            "utf-8",
            errors="replace"
        )

        raise RuntimeError(
            f"Blob HTTP错误 {exc.code}: "
            f"{detail[:1000]}"
        ) from exc

    except URLError as exc:

        raise RuntimeError(
            f"Blob网络错误：{exc}"
        ) from exc


# ============================================================
# 图片URL下载
# ============================================================

def download_image_url(url: str):

    request = Request(
        url,
        headers={
            "User-Agent":
                "Mozilla/5.0"
        }
    )

    try:

        with urlopen(
            request,
            timeout=60
        ) as response:

            data = response.read()

    except Exception as exc:

        raise RuntimeError(
            f"图片URL下载失败：{exc}"
        ) from exc

    if not data:

        raise RuntimeError(
            "图片下载为空。"
        )

    return data


# ============================================================
# 图片API
# ============================================================

def get_image_client():

    if not IMAGE_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="未配置 IMAGE_API_KEY。"
        )

    kwargs = {
        "api_key": IMAGE_API_KEY
    }

    if IMAGE_BASE_URL:
        kwargs["base_url"] = IMAGE_BASE_URL

    return OpenAI(**kwargs)


def generate_image_bytes(prompt: str):

    client = get_image_client()

    try:

        result = client.images.generate(
            model=IMAGE_MODEL,
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
            detail="图片API没有返回图片。"
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

    if b64:

        try:

            image_bytes = base64.b64decode(
                b64
            )

            print("✅ 图片获得 b64_json")

            return image_bytes

        except Exception as exc:

            raise HTTPException(
                status_code=502,
                detail=f"b64_json解析失败：{exc}"
            ) from exc

    if url:

        print("🌐 图片API返回URL，开始下载")

        return download_image_url(url)

    raise HTTPException(
        status_code=502,
        detail=(
            "无法识别图片API返回格式，"
            "既没有b64_json也没有url。"
        )
    )


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
            detail="找不到 frontend/index.html"
        )

    return FileResponse(
        index_path
    )


# ============================================================
# favicon
# ============================================================

@app.get("/favicon.ico")
def favicon():

    return Response(
        status_code=204
    )


@app.get("/favicon.png")
def favicon_png():

    return Response(
        status_code=204
    )


# ============================================================
# Health
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

    return {
        "status": "ok",
        "database": database_ok,
        "database_error": database_error,
        "llm_configured": bool(
            LLM_API_KEY
        ),
        "image_configured": bool(
            IMAGE_API_KEY
        ),
        "blob_configured": bool(
            BLOB_READ_WRITE_TOKEN
        ),
        "blob_store_configured": bool(
            BLOB_STORE_ID
        )
    }


# ============================================================
# 第一步：生成实验策略
#
# 重要：
# 这里只调用LLM，不调用图片API。
#
# 这样Vercel不会因为一次请求持续太久而 Failed to fetch。
# ============================================================

@app.post("/api/generate")
def generate(
    req: GenerateRequest
):

    experiment_id = (
        uuid.uuid4()
        .hex[:12]
    )

    matched, unmatched = choose_conditions(
        req.mft
    )

    print("=" * 60)
    print("🚀 开始生成AIGC公益广告实验")
    print("=" * 60)
    print(f"📌 实验ID：{experiment_id}")
    print(f"📌 主题：{req.topic}")
    print(f"📌 匹配条件：{matched}")
    print(f"📌 不匹配条件：{unmatched}")

    ai = call_llm(
        build_generation_prompt(
            req.topic,
            req.mft,
            matched,
            unmatched
        )
    )

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

    return {
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
            matched_item,

        "unmatched":
            unmatched_item
    }


# ============================================================
# 第二步：单张图片生成
#
# 每次只生成一张。
# 前端会同时请求两个条件。
# ============================================================

@app.post("/api/generate-image")
def generate_image_endpoint(
    req: GenerateImageRequest
):

    if req.condition not in (
        "matched",
        "unmatched"
    ):

        raise HTTPException(
            status_code=400,
            detail="condition必须是matched或unmatched。"
        )

    print(
        f"🎨 正在生成 "
        f"{req.condition} 图片..."
    )

    image_bytes = generate_image_bytes(
        req.image_prompt
    )

    print(
        f"📦 图片大小："
        f"{len(image_bytes) / 1024 / 1024:.2f} MB"
    )

    image_url = upload_image_to_blob(
        image_bytes,
        req.experiment_id,
        req.condition
    )

    print(
        f"✅ {req.condition} 图片已经上传到 Vercel Blob"
    )

    return {
        "ok": True,
        "experiment_id":
            req.experiment_id,
        "condition":
            req.condition,
        "image_url":
            image_url
    }


# ============================================================
# 第三步：保存实验
# ============================================================

@app.post("/api/finalize")
def finalize(
    req: FinalizeRequest
):

    ensure_db()

    conn = None

    try:

        conn = db()

        # ----------------------------------------------------
        # 防止重复提交
        # ----------------------------------------------------

        existing = conn.execute(
            """
            SELECT id
            FROM experiments
            WHERE id = %s
            """,
            (
                req.experiment_id,
            )
        ).fetchone()

        if existing:

            return {
                "ok": True,
                "experiment_id":
                    req.experiment_id,
                "already_exists":
                    True
            }

        now = datetime.now(
            timezone.utc
        )

        # ----------------------------------------------------
        # experiment
        # ----------------------------------------------------

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
                req.experiment_id,
                req.topic,
                req.mft.care,
                req.mft.fairness,
                req.mft.loyalty,
                req.mft.authority,
                req.mft.sanctity,
                req.matched_foundation,
                req.unmatched_foundation,
                now
            )
        )

        # ----------------------------------------------------
        # ads
        # ----------------------------------------------------

        for condition, item in [
            (
                "matched",
                req.matched
            ),
            (
                "unmatched",
                req.unmatched
            )
        ]:

            ad_id = (
                uuid.uuid4()
                .hex[:12]
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

            image_url = item.get(
                "image_url"
            )

            foundation = item.get(
                "foundation"
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
                    image_url,
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
                    %s
                )
                """,
                (
                    ad_id,
                    req.experiment_id,
                    condition,
                    foundation,
                    json.dumps(
                        strategy,
                        ensure_ascii=False
                    ),
                    copy_text,
                    image_prompt,
                    image_url,
                    now
                )
            )

            item["ad_id"] = ad_id

        conn.commit()

        print(
            f"🎉 实验保存成功："
            f"{req.experiment_id}"
        )

        return {
            "ok": True,
            "experiment_id":
                req.experiment_id,
            "matched":
                req.matched,
            "unmatched":
                req.unmatched
        }

    except Exception as exc:

        if conn:
            conn.rollback()

        raise HTTPException(
            status_code=500,
            detail=f"实验保存失败：{exc}"
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

    ensure_db()

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
            detail=f"评价保存失败：{exc}"
        ) from exc

    finally:

        if conn:
            conn.close()


# ============================================================
# 历史记录
# ============================================================

@app.get("/api/history")
def get_history():

    ensure_db()

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

            history.append(
                {
                    "experiment_id":
                        row["id"],

                    "topic":
                        row["topic"],

                    "mft":
                        {
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
                }
            )

        return {
            "total":
                len(history),
            "history":
                history
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"历史记录读取失败：{exc}"
        ) from exc

    finally:

        if conn:
            conn.close()


# ============================================================
# 历史详情
# ============================================================

@app.get(
    "/api/history/{experiment_id}"
)
def get_history_detail(
    experiment_id: str
):

    ensure_db()

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

            "mft":
                {
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

            result[
                "evaluations"
            ].append(
                {
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
                }
            )

        return result

    except HTTPException:

        raise

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"历史实验详情读取失败：{exc}"
        ) from exc

    finally:

        if conn:
            conn.close()


# ============================================================
# 本地运行
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(
            os.getenv(
                "PORT",
                "8000"
            )
        )
    )
