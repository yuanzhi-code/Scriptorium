import whisperx
import gc
import torch
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def transcribe_and_diarize_two_speakers(audio_file_path, output_txt_path, language=None, model_size="small"):
    """
    使用 WhisperX 转录音频，并明确指示只有两个发言人，然后将结果输出到文本文件。

    Args:
        audio_file_path (str): 输入音频文件的路径。
        output_txt_path (str): 输出文本文件的路径。
        language (str, optional): 音频语言，例如 "zh" (中文), "en" (英文)。
                                  如果为 None，WhisperX 将自动检测语言。
        model_size (str, optional): Whisper 模型大小。可选值如 "large-v3", "medium", "small", "base"。
                                   "large-v3" 提供最佳准确率，但速度慢且占用资源多。
    """
    # 检查 Hugging Face Token
    if not os.getenv("HF_TOKEN"):
        raise ValueError("请设置环境变量 HF_TOKEN。您可以通过以下方式设置：\n"
                        "1. 访问 https://huggingface.co/settings/tokens 创建 Token\n"
                        "2. 运行: export HF_TOKEN=your_token_here\n"
                        "3. 或者创建 .env 文件并添加: HF_TOKEN=your_token_here")

    # --- 1. 设置设备 ---
    # 优先使用 CUDA (NVIDIA GPU) -> MPS (Apple Silicon GPU) -> CPU
    if torch.cuda.is_available():
        device = "cuda"
        compute_type = "float16"
    elif torch.backends.mps.is_available():
        device = "mps"
        compute_type = "float16"
    else:
        device = "cpu"
        compute_type = "int8"

    print(f"Using device: {device}")
    print(f"Using compute type: {compute_type}")

    batch_size = 16 # 可以根据你的硬件资源调整

    # --- 2. 加载音频 ---
    print(f"Loading audio from: {audio_file_path}")
    audio = whisperx.load_audio(audio_file_path)

    # --- 3. 加载 Whisper 模型并转录 ---
    print(f"Loading Whisper model ({model_size})...")
    model = whisperx.load_model(model_size, device, compute_type=compute_type, language=language)

    print("Starting transcription...")
    result = model.transcribe(audio, batch_size=batch_size, word_timestamps=True)

    # --- 4. 加载对齐模型并进行对齐 ---
    print(f"Loading alignment model for language: {result['language']}...")
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    print("Aligning segments for accurate word-level timestamps...")
    result = whisperx.align(result["segments"], model_a, audio, device, return_char_alignments=False)

    # --- 5. 执行说话人区分 (明确指定两个发言人) ---
    print("Performing speaker diarization for **two speakers**...")
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=True, device=device)
    
    # *** 关键改动：在这里设置 min_speakers 和 max_speakers 为 2 ***
    diarize_segments = diarize_model(audio, min_speakers=2, max_speakers=2) 

    # --- 6. 将说话人标签分配到转录文本 ---
    print("Assigning speaker labels to transcription segments...")
    result = whisperx.assign_word_speakers(diarize_segments, result)

    # --- 7. 格式化输出并保存到文件 ---
    print(f"Saving transcription to {output_txt_path}...")
    with open(output_txt_path, "w", encoding="utf-8") as f:
        for segment in result["segments"]:
            speaker_label = segment.get("speaker", "Unknown")
            start_time = segment["start"]
            end_time = segment["end"]
            text = segment["text"].strip()

            f.write(f"[{speaker_label}] [{start_time:.2f}s - {end_time:.2f}s]: {text}\n")
            
    print("Transcription complete and saved!")

    # --- 8. 内存清理 (可选) ---
    del model
    del model_a
    del diarize_model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

# --- 如何使用 ---
if __name__ == "__main__":
    # 设置输入输出路径
    input_audio_file = "123.m4a"  # 替换为您的音频文件路径
    output_text_file = "output.txt"  # 替换为您想要保存的文本文件路径

    try:
        # 检查输入文件是否存在
        if not os.path.exists(input_audio_file):
            raise FileNotFoundError(f"找不到音频文件: {input_audio_file}")

        # 调用转录函数
        print(f"开始处理音频文件: {input_audio_file}")
        transcribe_and_diarize_two_speakers(
            audio_file_path=input_audio_file,
            output_txt_path=output_text_file,
            language="zh",  # 设置为中文
            model_size="small"  # 使用最高精度的模型
        )

        print(f"\n转录完成！请查看文件: {output_text_file}")
        print("文件格式说明：")
        print("- [SPEAKER_0] 表示第一个说话人")
        print("- [SPEAKER_1] 表示第二个说话人")
        print("- 时间戳格式为 [开始时间 - 结束时间]")
        
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")