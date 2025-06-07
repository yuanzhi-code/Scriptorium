# Scriptorium

院止的常用脚本合集，包含各种实用的音频处理、文本处理等工具。

## 项目结构

### WhisperX_chinese_2_speakers
中文双人对话转录工具，使用 WhisperX 实现高质量的语音转文字功能。
- 支持中文音频转录
- 自动识别两个说话人
- 生成带时间戳的转录文本
- 支持多种音频格式
- 使用 GPU 加速（如果可用）

[查看详细说明](./WhisperX_chinese_2_speakers/README.md)

## 环境要求

- Python 3.8 或更高版本
- uv 包管理器（推荐）
- 足够的磁盘空间用于存储模型

## 快速开始

1. 克隆仓库：
   ```bash
   git clone https://github.com/yourusername/Scriptorium.git
   cd Scriptorium
   ```

2. 选择需要的工具，进入对应目录：
   ```bash
   cd WhisperX_chinese_2_speakers  # 或其他工具目录
   ```

3. 按照各个工具的 README 说明进行安装和使用

## 工具列表

- [WhisperX 中文双人对话转录](./WhisperX_chinese_2_speakers/README.md)
  - 使用 WhisperX 进行中文语音转录
  - 支持说话人区分
  - 生成带时间戳的文本

## 贡献指南

欢迎提交 Pull Request 或创建 Issue 来改进这个项目。

## 许可证

请确保遵守各个工具的使用条款和许可证要求。
