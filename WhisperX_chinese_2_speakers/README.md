# WhisperX 对话转录

这个项目使用 WhisperX 来转录对话，并支持说话人区分和文本对齐功能。

## 功能特点

- 支持中文音频转录
- 自动识别两个说话人
- 生成带时间戳的转录文本
- 支持多种音频格式（自动转换为 WAV）
- 使用 GPU 加速（如果可用）
- 支持从中间结果继续处理
- 自动音频格式转换（支持 m4a、mp3 等格式）

## 环境要求

- Python 3.8 或更高版本
- CUDA 支持（可选，用于 GPU 加速）
- 足够的磁盘空间用于存储模型
- ffmpeg（用于音频格式转换）
- uv 包管理器（推荐）

## 模型存储位置

首次运行时，程序会自动下载所需的模型文件。模型文件存储在以下位置：

- Linux/Mac: `~/.cache/huggingface/hub/`
- Windows: `C:\Users\<username>\.cache\huggingface\hub\`

具体模型文件：
- Whisper 模型：`models--openai--whisper-<model_size>/`
- 对齐模型：`models--whisperx--alignment-model-<language>/`
- 说话人区分模型：`models--pyannote--speaker-diarization/`

注意：确保有足够的磁盘空间，所有模型文件总共需要约 15GB 空间。

## 快速开始

1. 克隆项目并进入项目目录：
   ```bash
   cd WhisperX_chinese_2_speakers
   ```

2. 使用 uv 创建虚拟环境并安装依赖：
   ```bash
   # 创建虚拟环境
   uv venv
   
   # 激活虚拟环境
   source .venv/bin/activate
   
   # 安装依赖（使用 lock 文件确保版本一致）
   uv pip install -r uv.lock
   ```

3. 设置 Hugging Face Token：
   - 访问 [Hugging Face Token 设置页面](https://huggingface.co/settings/tokens)
   - 创建一个新的 Token（确保选择 "Read" 权限）
   - 同意以下模型的使用条款：
     - [pyannote/speaker-diarization](https://huggingface.co/pyannote/speaker-diarization)
     - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   - 创建 `.env` 文件并添加您的 Token：
     ```bash
     echo "HF_TOKEN=your_token_here" > .env
     ```

## 使用方法

1. 准备您的音频文件（支持 wav、mp3、m4a 等格式）

2. 修改 `audio2text.py` 中的输入输出路径：
   ```python
   input_audio_file = "your_audio_file.m4a"  # 替换为您的音频文件路径
   output_text_file = "output.txt"  # 替换为您想要保存的文本文件路径
   ```

3. 运行转录脚本：
   ```bash
   python audio2text.py
   ```

## 高级配置选项

转录函数支持以下参数配置：

```python
transcribe_and_diarize_two_speakers(
    audio_file_path="input.m4a",      # 输入音频文件路径
    output_txt_path="output.txt",     # 输出文本文件路径
    language="zh",                    # 语言代码（zh 表示中文）
    model_size="small",               # 模型大小
    enable_alignment=False,           # 是否启用文本对齐
    enable_diarization=True,          # 是否启用说话人区分
    load_intermediate=False           # 是否从中间结果加载
)
```

## 模型大小选项

WhisperX 支持以下模型大小选项，您可以根据需求选择合适的模型：

| 模型大小 | 参数数量 | 说明 | 内存占用 | 速度 | 准确率 | 适用场景 |
|---------|---------|------|---------|------|--------|---------|
| `large-v3` | 1.5B | 最大模型，最新版本 | 约 10GB | 最慢 | 最高 | 专业转录、高精度要求 |
| `large-v2` | 1.5B | 大模型，第二版 | 约 10GB | 慢 | 高 | 专业转录、高精度要求 |
| `large` | 1.5B | 大模型，第一版 | 约 10GB | 慢 | 高 | 专业转录、高精度要求 |
| `medium` | 769M | 中等模型 | 约 5GB | 中等 | 中等 | 一般转录、平衡性能 |
| `small` | 244M | 小模型 | 约 1GB | 快 | 一般 | 普通对话、基本硬件 |
| `base` | 74M | 基础模型 | 约 1GB | 最快 | 基础 | 简单对话、资源受限 |

选择建议：
- 如果您的设备有足够的 GPU 内存（>= 12GB），建议使用 `large-v3`
- 如果 GPU 内存有限（8GB），可以使用 `medium`
- 如果使用 CPU 或内存受限：
  - 需要更好的准确率：选择 `small`（244M 参数）
  - 需要更快的速度：选择 `base`（74M 参数）

## 输出格式

转录结果将保存在指定的文本文件中，格式如下：
```
[SPEAKER_0] [开始时间 - 结束时间]: 说话内容
[SPEAKER_1] [开始时间 - 结束时间]: 说话内容
```

## 注意事项

1. 首次运行时会下载必要的模型，这可能需要一些时间
2. 确保有足够的磁盘空间（模型文件较大）
3. 如果使用 GPU，确保已正确安装 CUDA
4. 音频文件最好是清晰的对话录音，背景噪音会影响识别效果
5. 建议使用 uv 包管理器来管理依赖，以确保环境一致性

## 常见问题

1. 如果遇到 "CUDA not available" 错误：
   - 检查是否正确安装了 CUDA
   - 或者程序会自动使用 CPU 模式

2. 如果遇到 "HF_TOKEN not set" 错误：
   - 确保已正确设置 Hugging Face Token
   - 检查 `.env` 文件是否存在且包含正确的 Token

3. 如果转录效果不理想：
   - 尝试使用更清晰的音频文件
   - 确保音频文件格式正确
   - 考虑使用更大的模型（修改 `model_size` 参数）

4. 如果依赖安装失败：
   - 确保使用 uv 包管理器
   - 尝试使用 `uv.lock` 文件安装依赖
   - 检查 Python 版本是否符合要求

## 许可证

请确保遵守 WhisperX 和 pyannote.audio 的使用条款和许可证要求。 