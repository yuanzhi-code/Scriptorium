import whisperx
import gc
import torch
import os
from dotenv import load_dotenv
from pyannote.audio import Pipeline

# 加载环境变量
load_dotenv()

def transcribe_and_diarize_two_speakers(audio_file_path, output_txt_path, language=None, model_size="small", 
                                      enable_alignment=False, enable_diarization=True):
    """
    使用 WhisperX 转录音频，并明确指示只有两个发言人，然后将结果输出到文本文件。

    Args:
        audio_file_path (str): 输入音频文件的路径。
        output_txt_path (str): 输出文本文件的路径。
        language (str, optional): 音频语言，例如 "zh" (中文), "en" (英文)。
                                  如果为 None，WhisperX 将自动检测语言。
        model_size (str, optional): Whisper 模型大小。可选值如 "large-v3", "medium", "small", "base"。
                                   "large-v3" 提供最佳准确率，但速度慢且占用资源多。
        enable_alignment (bool, optional): 是否启用文本对齐功能。默认为 False。
        enable_diarization (bool, optional): 是否启用说话人区分功能。默认为 True。
    """
    # 检查 Hugging Face Token
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("请设置环境变量 HF_TOKEN。您可以通过以下方式设置：\n"
                        "1. 访问 https://huggingface.co/settings/tokens 创建 Token\n"
                        "2. 运行: export HF_TOKEN=your_token_here\n"
                        "3. 或者创建 .env 文件并添加: HF_TOKEN=your_token_here")
    
    if enable_diarization:
        print("\n使用说话人区分功能需要额外的设置：")
        print("1. 访问 https://huggingface.co/pyannote/speaker-diarization 并接受使用条款")
        print("2. 访问 https://huggingface.co/pyannote/segmentation 并接受使用条款")
        print("3. 确保您的 HF_TOKEN 有正确的权限\n")

    # --- 1. 设置设备 ---
    # 优先使用 CUDA (NVIDIA GPU) -> CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        compute_type = "float16"
    else:
        device = torch.device("cpu")
        compute_type = "int8"  # CPU 模式下使用 int8

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
    result = model.transcribe(audio, batch_size=batch_size)

    # 保存中间结果（纯文本）
    intermediate_output = output_txt_path.replace(".txt", "_intermediate.txt")
    print(f"Saving intermediate transcription to {intermediate_output}...")
    with open(intermediate_output, "w", encoding="utf-8") as f:
        for segment in result["segments"]:
            text = segment["text"].strip()
            f.write(f"{text}\n")
    print("Intermediate transcription saved!")

    # --- 4. 加载对齐模型并进行对齐（如果启用） ---
    if enable_alignment:
        print(f"Loading alignment model for language: {result['language']}...")
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        print("Aligning segments for accurate word-level timestamps...")
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    # --- 5. 执行说话人区分（如果启用） ---
    if enable_diarization:
        try:
            print("Performing speaker diarization for **two speakers**...")
            diarize_model = Pipeline.from_pretrained(
                "pyannote/speaker-diarization",
                use_auth_token=hf_token
            ).to(device)
            diarize_segments = diarize_model(audio, min_speakers=2, max_speakers=2)
            print("Assigning speaker labels to transcription segments...")
            result = whisperx.assign_word_speakers(diarize_segments, result)
        except Exception as e:
            print("\n说话人区分失败，可能的原因：")
            print("1. 您还没有接受模型的使用条款")
            print("2. HF_TOKEN 权限不足")
            print("3. 网络连接问题")
            print("\n您可以：")
            print("1. 重新运行程序，但禁用说话人区分（设置 enable_diarization=False）")
            print("2. 检查并接受模型使用条款")
            print("3. 重新生成 HF_TOKEN 并确保有正确的权限")
            raise Exception(f"说话人区分失败: {str(e)}")

    # --- 6. 格式化输出并保存到文件 ---
    print(f"Saving transcription to {output_txt_path}...")
    with open(output_txt_path, "w", encoding="utf-8") as f:
        for segment in result["segments"]:
            if enable_diarization:
                speaker_label = segment.get("speaker", "Unknown")
                speaker_prefix = f"[{speaker_label}] "
            else:
                speaker_prefix = ""

            if enable_alignment:
                start_time = segment["start"]
                end_time = segment["end"]
                time_prefix = f"[{start_time:.2f}s - {end_time:.2f}s] "
            else:
                time_prefix = ""

            text = segment["text"].strip()
            f.write(f"{speaker_prefix}{time_prefix}{text}\n")
            
    print("Transcription complete and saved!")

    # --- 7. 内存清理 (可选) ---
    del model
    if enable_alignment:
        del model_a
    if enable_diarization:
        del diarize_model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

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
            model_size="small",  # 使用最高精度的模型
            enable_alignment=False,  # 禁用对齐
            enable_diarization=True  # 启用说话人区分
        )

        print(f"\n转录完成！请查看文件: {output_text_file}")
        if enable_diarization:
            print("文件格式说明：")
            print("- [SPEAKER_0] 表示第一个说话人")
            print("- [SPEAKER_1] 表示第二个说话人")
        
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")