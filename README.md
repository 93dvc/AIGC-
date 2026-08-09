# AIGC 个性化公益广告实验平台 MVP

当前版本：
- 用户人工输入 MFT 五大维度 0-10 分
- 用户人工输入公益广告主题
- 最高分维度自动作为匹配条件
- 最低分维度自动作为不匹配条件
- LLM 生成两套传播策略、文案和图片 Prompt
- Image API 生成两张海报
- SQLite 保存实验与评价
- API Key 只在后端 `.env`

## Windows 启动

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
uvicorn backend.main:app --reload
```

浏览器：
`http://127.0.0.1:8000`

## API 配置

`.env` 示例：

```env
LLM_API_KEY=你的文本模型API_KEY
LLM_BASE_URL=你的文本模型兼容接口/v1
LLM_MODEL=你的文本模型

IMAGE_API_KEY=你的生图API_KEY
IMAGE_BASE_URL=你的生图接口/v1
IMAGE_MODEL=gpt-image-2
```

你之前已经测试成功的图片中转接口可以直接配置到 `IMAGE_BASE_URL`。

## 当前实验逻辑

例如：

Care=8
Fairness=5
Loyalty=4
Authority=3
Sanctity=5

主题：
海洋塑料污染

自动得到：
- Match：Care / Harm
- Unmatch：Authority / Subversion

之后由 AI 依次完成：
MFT画像 → 传播策略 → 文案 → 图片Prompt → 图片生成

正式实验时应隐藏匹配标签并随机化 A/B 顺序；当前 MVP 为了方便开发调试会显示条件。
