from pathlib import Path
import base64
import json
import os
import uuid
from datetime import datetime, timezone
from urllib.request import Request, urlopen
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from openai import OpenAI
from pydantic import BaseModel, Field

import psycopg
from psycopg.rows import dict_row

from vercel.blob import BlobClient


# ============================================================
# 基础路径
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="AIGC 个性化公益广告实验平台",
    version="2.0.0"
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
# PostgreSQL
# ============================================================

DATABASE_URL = os.getenv("DATABASE_URL")


def db():
    """
    PostgreSQL 数据库连接。
    """

    database_url = os.getenv("DATABASE_URL")

    if not database_url:

        raise RuntimeError(
            "未配置 DATABASE_URL。"
            "请在 Vercel Environment Variables 中配置。"
        )

    return psycopg.connect(
        database_url,
        row_factory=dict_row
    )


# ============================================================
# 初始化数据库
# ============================================================

def init_db():

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

                care DOUBLE PRECISION NOT NULL,

                fairness DOUBLE PRECISION NOT NULL,

                loyalty DOUBLE PRECISION NOT NULL,

                authority DOUBLE PRECISION NOT NULL,

                sanctity DOUBLE PRECISION NOT NULL,

                matched_foundation TEXT NOT NULL,

                unmatched_foundation TEXT NOT NULL,

                status TEXT NOT NULL DEFAULT 'pending',

                current_stage TEXT NOT NULL DEFAULT 'pending_llm',

                progress INTEGER NOT NULL DEFAULT 0,

                error_message TEXT,

                created_at TIMESTAMPTZ NOT NULL,

                updated_at TIMESTAMPTZ NOT NULL

            )
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

                strategy_json TEXT NOT NULL,

                copy TEXT NOT NULL,

                image_prompt TEXT NOT NULL,

                image_url TEXT,

                created_at TIMESTAMPTZ NOT NULL,

                FOREIGN KEY(experiment_id)
                    REFERENCES experiments(id)
                    ON DELETE CASCADE

            )
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

        # ----------------------------------------------------
        # 兼容旧数据库
        # ----------------------------------------------------

        existing_columns = conn.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'experiments'
            """
        ).fetchall()

        column_names = {
            row["column_name"]
            for row in existing_columns
        }

        if "status" not in column_names:

            conn.execute(
                """
                ALTER TABLE experiments
                ADD COLUMN status TEXT NOT NULL
                DEFAULT 'pending'
                """
            )

        if "current_stage" not in column_names:

            conn.execute(
                """
                ALTER TABLE experiments
                ADD COLUMN current_stage TEXT NOT NULL
                DEFAULT 'pending_llm'
                """
            )

        if "progress" not in column_names:

            conn.execute(
                """
                ALTER TABLE experiments
                ADD COLUMN progress INTEGER NOT NULL
                DEFAULT 0
                """
            )

        if "error_message" not in column_names:

            conn.execute(
                """
                ALTER TABLE experiments
                ADD COLUMN error_message TEXT
                """
            )

        if "updated_at" not in column_names:

            conn.execute(
                """
                ALTER TABLE experiments
                ADD COLUMN updated_at TIMESTAMPTZ
                """
            )

            conn.execute(
                """
                UPDATE experiments
                SET updated_at = created_at
                WHERE updated_at IS NULL
                """
            )

        conn.commit()

        print("✅ PostgreSQL 数据库初始化成功")

    except Exception:

        conn.rollback()

        raise

    finally:

        conn.close()


# ============================================================
# Vercel Serverless：
# 初始化失败不能让整个 import 崩掉
# ============================================================

try:

    init_db()

except Exception as exc:

    print(
        f"⚠️ 数据库初始化失败：{exc}"
    )


# ============================================================
# Pydantic
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
# 工具函数
# ============================================================

def now_utc():

    return datetime.now(
        timezone.utc
    )


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
# API Client
# ============================================================

def get_client(
    api_key_name: str,
    base_url_name: str
):

    key = os.getenv(
        api_key_name
    )

    base_url = os.getenv(
        base_url_name
    )

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

    return OpenAI(
        **kwargs
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

    scores = scores_dict(
        mft
    )

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

必须尽可能改变：

1. 核心主体
2. 场景
3. 视觉隐喻
4. 构图
5. 情绪
6. 视觉符号
7. 摄影角度
8. 文字排版

但是：

两张海报必须拥有完全相同的公益主题。

匹配版必须围绕：

{FOUNDATIONS[matched]['cn']}

展开。

不匹配版必须围绕：

{FOUNDATIONS[unmatched]['cn']}

展开。

MFT参考：

Care / Harm：

强调生命、保护、伤害、脆弱、陪伴、救助。

Fairness / Cheating：

强调公平、不公平、交换、失衡、规则、机会差距。

Loyalty / Betrayal：

强调共同体、承诺、责任、关系、背叛、共同守护。

Authority / Subversion：

强调规则、秩序、责任、公共规范、社会制度。

Sanctity / Degradation：

强调纯净、污染、神圣、洁净与肮脏之间的冲突。

必须生成两句中文公益广告标语。

要求：

1. 完全相同的公益主题。
2. 明显的对仗关系。
3. 句式长度尽量接近。
4. 结构尽量对应。
5. 不能使用学术术语。
6. 必须自然。
7. 必须有传播性和记忆点。
8. 必须体现不同MFT道德价值。

例如：

匹配：

“少一份伤害，多一条生命的路。”

不匹配：

“守一份规则，多一片海洋的净土。”

这是海报生成任务。

图片中必须直接出现对应的中文公益广告标语。

禁止把标语留给网页后期叠加。

image_prompt必须明确写出：

EXACT CHINESE SLOGAN TO RENDER:

“这里放copy中的完整中文标语”

图像模型必须尝试将这句话逐字、完整、清晰地生成在海报中。

不得修改文字。

不得增加额外文字。

不得删除文字。

不得生成英文翻译。

不得生成随机英文。

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

标语位置不限。

但必须：

清晰、完整、可读。

允许：

对应的中文公益广告标语。

禁止：

Logo
水印
品牌名称
网址
其他无关文字
随机英文
随机字母
额外标语

匹配版和不匹配版必须视觉概念明显不同。

不要只是改变颜色。

不要只是改变文字。

不要只是改变一个主体。

每个image_prompt必须是完整英文Prompt。

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

不要Markdown。
不要代码块。
"""


# ============================================================
# LLM
# ============================================================

def call_llm(prompt: str):

    print("🤖 开始调用 LLM...")

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

        raise RuntimeError(
            f"LLM调用失败：{exc}"
        ) from exc

    if not response.choices:

        raise RuntimeError(
            "LLM没有返回choices。"
        )

    content = (
        response
        .choices[0]
        .message
        .content
    )

    if not content:

        raise RuntimeError(
            "LLM返回内容为空。"
        )

    content = content.strip()

    try:

        result = json.loads(
            content
        )

    except json.JSONDecodeError as exc:

        raise RuntimeError(
            "LLM返回内容不是有效JSON。"
            f"实际返回长度：{len(content)}。"
            f"返回开头：{content[:500]}"
        ) from exc

    return result


# ============================================================
# Blob 上传
# ============================================================

def upload_image_to_blob(
    image_bytes: bytes,
    experiment_id: str,
    condition: str
):

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
        f"☁️ 开始上传 Vercel Blob：{filename}"
    )

    try:

        token = os.getenv(
            "BLOB_READ_WRITE_TOKEN"
        )

        if not token:

            token = os.getenv(
                "VERCEL_BLOB_READ_WRITE_TOKEN"
            )

        if not token:

            raise RuntimeError(
                "没有找到 BLOB_READ_WRITE_TOKEN。"
            )

        client = BlobClient(
            token=token
        )

        blob = client.put(

            filename,

            image_bytes,

            access="public",

            content_type="image/png",

            add_random_suffix=True

        )

        if not blob:

            raise RuntimeError(
                "Blob API没有返回对象。"
            )

        blob_url = getattr(
            blob,
            "url",
            None
        )

        if not blob_url:

            raise RuntimeError(
                "Blob上传成功但没有返回URL。"
            )

        print(
            f"✅ Vercel Blob 上传成功："
            f"{blob_url}"
        )

        return blob_url

    except Exception as exc:

        print(
            f"❌ Blob上传失败：{exc}"
        )

        raise RuntimeError(
            f"Vercel Blob 上传失败：{exc}"
        ) from exc


# ============================================================
# 下载图片URL
# ============================================================

def download_image_url(
    url: str
):

    try:

        request = Request(

            url,

            headers={
                "User-Agent":
                    "Mozilla/5.0"
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

    print(
        f"🎨 正在生成 {condition} 图片..."
    )

    client = get_client(
        "IMAGE_API_KEY",
        "IMAGE_BASE_URL"
    )

    model = os.getenv(
        "IMAGE_MODEL",
        "gpt-image-2"
    )

    try:

        result = client.images.generate(

            model=model,

            prompt=prompt,

            size="1024x1536",

            n=1

        )

    except Exception as exc:

        raise RuntimeError(
            f"{condition} 图片API调用失败：{exc}"
        ) from exc

    if not result.data:

        raise RuntimeError(
            f"{condition} 图片API没有返回数据。"
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

    # --------------------------------------------------------
    # b64
    # --------------------------------------------------------

    if b64:

        try:

            image_bytes = base64.b64decode(
                b64
            )

            print(
                f"✅ {condition} 图片获得 b64_json"
            )

        except Exception as exc:

            print(
                f"⚠️ b64_json解析失败：{exc}"
            )

    # --------------------------------------------------------
    # URL
    # --------------------------------------------------------

    if image_bytes is None and url:

        print(
            f"🌐 {condition} 图片获得URL"
        )

        image_bytes = download_image_url(
            url
        )

    if image_bytes is None:

        raise RuntimeError(
            f"{condition} 图片API没有返回"
            "b64_json或url。"
        )

    # --------------------------------------------------------
    # Blob
    # --------------------------------------------------------

    blob_url = upload_image_to_blob(

        image_bytes,

        experiment_id,

        condition

    )

    return {
        "url": blob_url,
        "source": "vercel_blob"
    }


# ============================================================
# 创建实验
# ============================================================

@app.post("/api/generate")
def create_experiment(
    req: GenerateRequest
):

    experiment_id = (
        uuid.uuid4()
        .hex[:12]
    )

    matched, unmatched = choose_conditions(
        req.mft
    )

    current = now_utc()

    print(
        "\n" +
        "=" * 60
    )

    print(
        "🚀 创建AIGC公益广告实验"
    )

    print(
        "=" * 60
    )

    print(
        f"📌 实验ID：{experiment_id}"
    )

    print(
        f"📌 主题：{req.topic}"
    )

    print(
        f"📌 匹配条件：{matched}"
    )

    print(
        f"📌 不匹配条件：{unmatched}"
    )

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
                status,
                current_stage,
                progress,
                error_message,
                created_at,
                updated_at
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

                "processing",

                "pending_llm",

                5,

                None,

                current,

                current

            )

        )

        conn.commit()

    except Exception as exc:

        if conn:
            conn.rollback()

        raise HTTPException(

            status_code=500,

            detail=(
                f"创建实验失败：{exc}"
            )

        )

    finally:

        if conn:
            conn.close()

    return {

        "ok": True,

        "experiment_id":
            experiment_id,

        "status":
            "processing",

        "stage":
            "pending_llm",

        "progress":
            5

    }


# ============================================================
# 状态辅助
# ============================================================

def get_experiment(
    experiment_id: str
):

    conn = db()

    try:

        row = conn.execute(

            """
            SELECT *
            FROM experiments
            WHERE id = %s
            """,

            (
                experiment_id,
            )

        ).fetchone()

        return row

    finally:

        conn.close()


def update_stage(
    experiment_id: str,
    stage: str,
    progress: int,
    status: str = "processing",
    error_message=None
):

    conn = db()

    try:

        conn.execute(

            """
            UPDATE experiments
            SET
                current_stage = %s,
                progress = %s,
                status = %s,
                error_message = %s,
                updated_at = %s
            WHERE id = %s
            """,

            (

                stage,

                progress,

                status,

                error_message,

                now_utc(),

                experiment_id

            )

        )

        conn.commit()

    finally:

        conn.close()


# ============================================================
# 保存广告
# ============================================================

def save_ad(
    experiment_id: str,
    condition: str,
    foundation: str,
    item: dict,
    image_url: str
):

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

    conn = db()

    try:

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

                experiment_id,

                condition,

                foundation,

                json.dumps(
                    strategy,
                    ensure_ascii=False
                ),

                copy_text,

                image_prompt,

                image_url,

                now_utc()

            )

        )

        conn.commit()

    finally:

        conn.close()

    return ad_id


# ============================================================
# 获取实验结果
# ============================================================

def build_experiment_result(
    experiment_id: str
):

    conn = db()

    try:

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

            return None

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

            "status":
                experiment["status"],

            "stage":
                experiment["current_stage"],

            "progress":
                experiment["progress"],

            "error":
                experiment["error_message"],

            "matched":
                None,

            "unmatched":
                None

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

        return result

    finally:

        conn.close()


# ============================================================
# 核心：轮询式状态机
# ============================================================

@app.get(
    "/api/generate/status/{experiment_id}"
)
def generate_status(
    experiment_id: str
):

    experiment = get_experiment(
        experiment_id
    )

    if not experiment:

        raise HTTPException(
            status_code=404,
            detail="找不到实验。"
        )

    status = experiment[
        "status"
    ]

    stage = experiment[
        "current_stage"
    ]

    # ========================================================
    # 已完成
    # ========================================================

    if status == "completed":

        return build_experiment_result(
            experiment_id
        )

    # ========================================================
    # 已失败
    # ========================================================

    if status == "failed":

        return {

            "experiment_id":
                experiment_id,

            "status":
                "failed",

            "stage":
                stage,

            "progress":
                experiment["progress"],

            "error":
                experiment["error_message"]

        }

    # ========================================================
    # 第一步：LLM
    # ========================================================

    if stage == "pending_llm":

        update_stage(

            experiment_id,

            "llm_processing",

            10

        )

        try:

            mft = MFTScores(

                care=experiment["care"],

                fairness=experiment["fairness"],

                loyalty=experiment["loyalty"],

                authority=experiment["authority"],

                sanctity=experiment["sanctity"]

            )

            matched = experiment[
                "matched_foundation"
            ]

            unmatched = experiment[
                "unmatched_foundation"
            ]

            prompt = build_generation_prompt(

                experiment["topic"],

                mft,

                matched,

                unmatched

            )

            ai = call_llm(
                prompt
            )

            matched_item = ai.get(
                "matched"
            )

            unmatched_item = ai.get(
                "unmatched"
            )

            if not matched_item:

                raise RuntimeError(
                    "LLM没有返回matched。"
                )

            if not unmatched_item:

                raise RuntimeError(
                    "LLM没有返回unmatched。"
                )

            if not matched_item.get(
                "image_prompt"
            ):

                raise RuntimeError(
                    "matched缺少image_prompt。"
                )

            if not unmatched_item.get(
                "image_prompt"
            ):

                raise RuntimeError(
                    "unmatched缺少image_prompt。"
                )

            # 暂时把策略存入临时字段不方便，
            # 所以直接创建广告记录。
            conn = db()

            try:

                # 防止重复创建
                existing = conn.execute(

                    """
                    SELECT COUNT(*) AS count
                    FROM ads
                    WHERE experiment_id = %s
                    """,

                    (
                        experiment_id,
                    )

                ).fetchone()

                if existing["count"] == 0:

                    for condition, foundation in [

                        (
                            "matched",
                            matched
                        ),

                        (
                            "unmatched",
                            unmatched
                        )

                    ]:

                        item = ai[
                            condition
                        ]

                        ad_id = (
                            uuid.uuid4()
                            .hex[:12]
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

                                experiment_id,

                                condition,

                                foundation,

                                json.dumps(
                                    item.get(
                                        "strategy",
                                        {}
                                    ),
                                    ensure_ascii=False
                                ),

                                item.get(
                                    "copy",
                                    ""
                                ),

                                item.get(
                                    "image_prompt",
                                    ""
                                ),

                                None,

                                now_utc()

                            )

                        )

                conn.commit()

            finally:

                conn.close()

            update_stage(

                experiment_id,

                "pending_matched_image",

                30

            )

            return {

                "experiment_id":
                    experiment_id,

                "status":
                    "processing",

                "stage":
                    "pending_matched_image",

                "progress":
                    30,

                "message":
                    "LLM策略生成完成，准备生成匹配海报。"

            }

        except Exception as exc:

            print(
                f"❌ LLM阶段失败：{exc}"
            )

            update_stage(

                experiment_id,

                "llm_failed",

                10,

                "failed",

                str(exc)

            )

            return {

                "experiment_id":
                    experiment_id,

                "status":
                    "failed",

                "stage":
                    "llm_failed",

                "progress":
                    10,

                "error":
                    str(exc)

            }

    # ========================================================
    # 第二步：matched
    # ========================================================

    if stage == "pending_matched_image":

        update_stage(

            experiment_id,

            "matched_image_processing",

            40

        )

        try:

            conn = db()

            try:

                ad = conn.execute(

                    """
                    SELECT *
                    FROM ads
                    WHERE experiment_id = %s
                    AND condition = 'matched'
                    LIMIT 1
                    """,

                    (
                        experiment_id,
                    )

                ).fetchone()

            finally:

                conn.close()

            if not ad:

                raise RuntimeError(
                    "找不到matched广告记录。"
                )

            if ad["image_url"]:

                update_stage(

                    experiment_id,

                    "pending_unmatched_image",

                    60

                )

                return {

                    "experiment_id":
                        experiment_id,

                    "status":
                        "processing",

                    "stage":
                        "pending_unmatched_image",

                    "progress":
                        60,

                    "message":
                        "matched图片已经存在。"

                }

            image_result = generate_image(

                ad["image_prompt"],

                experiment_id,

                "matched"

            )

            conn = db()

            try:

                conn.execute(

                    """
                    UPDATE ads
                    SET image_url = %s
                    WHERE id = %s
                    """,

                    (

                        image_result["url"],

                        ad["id"]

                    )

                )

                conn.commit()

            finally:

                conn.close()

            update_stage(

                experiment_id,

                "pending_unmatched_image",

                60

            )

            return {

                "experiment_id":
                    experiment_id,

                "status":
                    "processing",

                "stage":
                    "pending_unmatched_image",

                "progress":
                    60,

                "message":
                    "matched海报生成完成。"

            }

        except Exception as exc:

            print(
                f"❌ matched图片阶段失败：{exc}"
            )

            update_stage(

                experiment_id,

                "matched_image_failed",

                40,

                "failed",

                str(exc)

            )

            return {

                "experiment_id":
                    experiment_id,

                "status":
                    "failed",

                "stage":
                    "matched_image_failed",

                "progress":
                    40,

                "error":
                    str(exc)

            }

    # ========================================================
    # 第三步：unmatched
    # ========================================================

    if stage == "pending_unmatched_image":

        update_stage(

            experiment_id,

            "unmatched_image_processing",

            70

        )

        try:

            conn = db()

            try:

                ad = conn.execute(

                    """
                    SELECT *
                    FROM ads
                    WHERE experiment_id = %s
                    AND condition = 'unmatched'
                    LIMIT 1
                    """,

                    (
                        experiment_id,
                    )

                ).fetchone()

            finally:

                conn.close()

            if not ad:

                raise RuntimeError(
                    "找不到unmatched广告记录。"
                )

            if ad["image_url"]:

                update_stage(

                    experiment_id,

                    "completed",

                    100,

                    "completed"

                )

                return build_experiment_result(
                    experiment_id
                )

            image_result = generate_image(

                ad["image_prompt"],

                experiment_id,

                "unmatched"

            )

            conn = db()

            try:

                conn.execute(

                    """
                    UPDATE ads
                    SET image_url = %s
                    WHERE id = %s
                    """,

                    (

                        image_result["url"],

                        ad["id"]

                    )

                )

                conn.commit()

            finally:

                conn.close()

            update_stage(

                experiment_id,

                "completed",

                100,

                "completed"

            )

            print(
                f"🎉 实验完成：{experiment_id}"
            )

            return build_experiment_result(
                experiment_id
            )

        except Exception as exc:

            print(
                f"❌ unmatched图片阶段失败：{exc}"
            )

            update_stage(

                experiment_id,

                "unmatched_image_failed",

                70,

                "failed",

                str(exc)

            )

            return {

                "experiment_id":
                    experiment_id,

                "status":
                    "failed",

                "stage":
                    "unmatched_image_failed",

                "progress":
                    70,

                "error":
                    str(exc)

            }

    # ========================================================
    # 正在处理
    # ========================================================

    return {

        "experiment_id":
            experiment_id,

        "status":
            status,

        "stage":
            stage,

        "progress":
            experiment["progress"],

        "message":
            "任务正在处理中，请稍候。"

    }


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

    blob_token = (
        os.getenv(
            "BLOB_READ_WRITE_TOKEN"
        )
        or
        os.getenv(
            "VERCEL_BLOB_READ_WRITE_TOKEN"
        )
    )

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

        "image_configured":
            bool(
                os.getenv(
                    "IMAGE_API_KEY"
                )
            ),

        "blob_configured":
            bool(blob_token),

        "llm_base_url":
            bool(
                os.getenv(
                    "LLM_BASE_URL"
                )
            ),

        "image_base_url":
            bool(
                os.getenv(
                    "IMAGE_BASE_URL"
                )
            )

    }


# ============================================================
# 评价
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

                now_utc()

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

        )

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

                e.status,

                e.current_stage,

                e.progress,

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

                e.status,

                e.current_stage,

                e.progress,

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

                "status":
                    row["status"],

                "stage":
                    row["current_stage"],

                "progress":
                    row["progress"],

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

        )

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

            "status":
                experiment["status"],

            "stage":
                experiment["current_stage"],

            "progress":
                experiment["progress"],

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

        )

    finally:

        if conn:
            conn.close()


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
            )

        )

    return FileResponse(
        index_path
    )


# ============================================================
# favicon
# ============================================================

@app.get("/favicon.ico")
def favicon():

    return {
        "ok": True
    }


@app.get("/favicon.png")
def favicon_png():

    return {
        "ok": True
    }
