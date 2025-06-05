import sys
import os
import torch
import torchaudio
from einops import rearrange
# from stable_audio_tools import get_pretrained_model
# from stable_audio_tools.inference.generation import generate_diffusion_cond
import soundfile as sf 
import pyloudnorm as pyln
import numpy as np
from utils import LOUDNESS_NORM
BASE_DIR = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "TTS", "CosyVoice"))
sys.path.append(os.path.join(BASE_DIR, 'TTS', 'CosyVoice', 'third_party', 'Matcha-TTS'))
from cosyvoice.cli.cosyvoice import CosyVoice2
import logging
from pathlib import Path
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用的设备: {device}")

def load_cosyvoice_model(model_path='TTS/CosyVoice/pretrained_models/CosyVoice2-0.5B'):
    print("[模型加载] 正在加载 CosyVoice2 模型...")
    try:
        cosyvoice = CosyVoice2(
            os.path.join(BASE_DIR, model_path),
            load_jit=False,
            load_trt=False,
            fp16=False,
            use_flow_cache=False
        )
        print("[模型加载] CosyVoice2 模型加载完成。")
        return cosyvoice
    except Exception as e:
        print(f"[错误] CosyVoice2 模型加载失败: {e}")
        return None

def load_mmaudio_model(variant='large_44k_v2', full_precision=False):
    if variant not in all_model_cfg:
        raise ValueError(f'Unknown model variant: {variant}')
    
    model: ModelConfig = all_model_cfg[variant]
    model.download_if_needed()
    seq_cfg = model.seq_cfg

    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    dtype = torch.float32 if full_precision else torch.bfloat16

    net: MMAudio = get_my_mmaudio(model.model_name).to(device, dtype).eval()
    net.load_weights(torch.load(model.model_path, map_location=device, weights_only=True))

    feature_utils = FeaturesUtils(
        tod_vae_ckpt=model.vae_path,
        synchformer_ckpt=model.synchformer_ckpt,
        enable_conditions=True,
        mode=model.mode,
        bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
        need_vae_encoder=False
    ).to(device, dtype).eval()

    return {
        'model_cfg': model,
        'seq_cfg': seq_cfg,
        'net': net,
        'feature_utils': feature_utils,
        'device': device,
        'dtype': dtype
    }

def load_stable_audio_model(model_name="stabilityai/stable-audio-open-1.0"):
    print("[模型加载] 正在加载 Stable Audio 模型...")
    try:
        model, model_config = get_pretrained_model(model_name)
        model = model.to(device)
        sample_rate_val = model_config["sample_rate"]
        sample_size_val = model_config["sample_size"]
        print("[模型加载] Stable Audio 模型加载完成。")
        return model, model_config, sample_rate_val, sample_size_val
    except Exception as e:
        print(f"[错误] Stable Audio 模型加载失败: {e}")
        return None, None, None, None


def audio(model, out_wav="output_stable_audio.wav", text="128 BPM tech house drum loop",
                          length=30, current_sample_rate=44100, current_sample_size=None,
                          normalize=True, volume=-23.0, peak_norm_db_for_norm=-1.0):


    # print(f"[Stable Audio] 开始生成音频，提示: '{prompt}', 时长: {duration}s")
    conditioning = [{
        "prompt": text,
        "seconds_start": 0,
        "seconds_total": length
    }]
    sample_size= int(current_sample_rate * length)
    output_tensor = generate_diffusion_cond(
        model,
        steps=100,
        cfg_scale=7,
        conditioning=conditioning,
        sample_size=sample_size,
        sigma_min=0.3,
        sigma_max=500,
        sampler_type="dpmpp-3m-sde",
        device=device
    )
    # output_tensor 形状通常是 (batch_size, channels, samples)
    # 我们这里 batch_size=1, 所以是 (1, channels, samples)
    # rearrange to (channels, samples) for processing
    output_tensor = rearrange(output_tensor, "b d n -> d (b n)") # d is channels, n is samples per batch item

    if normalize:
        # print(f"[Stable Audio] 对 {output_path} 进行响度归一化...")
        # Stable Audio 的输出已经是 float32，范围 [-1, 1] (理论上)
        audio_np = output_tensor.cpu().numpy() # (channels, samples)
        
        # LOUDNESS_NORM(audio_data: np.ndarray, sr, target_lufs, peak_norm_db)
        normalized_audio_np = LOUDNESS_NORM(audio_np, sr=current_sample_rate,
                                            target_lufs=volume, peak_norm_db=peak_norm_db_for_norm)
        
        # 转换回 PyTorch tensor 以便 torchaudio.save 保存为 int16
        final_output_tensor = torch.from_numpy(normalized_audio_np).to(torch.float32)
        # torchaudio.save 会处理从 float32 [-1,1] 到 int16 的转换
        # print(f"[Stable Audio] 响度归一化完成，保存音频: {output_path}")
    else:
        final_output_tensor = output_tensor.cpu()
        print(f"[Stable Audio] 未进行响度归一化，保存音频: {out_wav}")

    # 确保最终张量在 CPU 上，并且进行适当的类型转换和缩放以保存为 int16
    # torchaudio.save 期望 float tensor 在 [-1, 1] 范围内来正确转换为 int16
    # 如果 LOUDNESS_NORM 的输出可能超出此范围（尽管不太可能），需要 clamp
    final_output_tensor = final_output_tensor.clamp(-1, 1)
    torchaudio.save(out_wav, final_output_tensor, current_sample_rate)
    print(f"[Stable Audio] 音频已保存: {out_wav}")


def tts(model, text, speaker_id, out_wav="output_cosyvoice.wav", speed=1.0,
        normalize=True, volume=-23.0, peak_norm_db_for_norm=-1.0):
    if model is None:
        print("[CosyVoice2] 模型未加载，跳过生成。")
        return

    print(f"[CosyVoice2] 开始生成TTS，文本: '{text[:30]}...', 情感: {speaker_id}")
    
    try:
        all_audio = []

        for i, j in enumerate(model.inference_zero_shot_lm(
                text, speaker_id, stream=False, speed=speed, text_frontend=True)):
            
            speech_tensor = j['tts_speech']  # (1, samples)

            if normalize:
                audio_np = speech_tensor.cpu().numpy()  # (1, samples)
                normalized_audio_np = LOUDNESS_NORM(audio_np, sr=model.sample_rate,
                                                    target_lufs=volume, peak_norm_db=peak_norm_db_for_norm)
                final_tensor = torch.from_numpy(normalized_audio_np).to(torch.float32)
                print(f"[CosyVoice2] [{i}] 响度归一化完成")
            else:
                final_tensor = speech_tensor.cpu()
                print(f"[CosyVoice2] [{i}] 未进行响度归一化")

            final_tensor = final_tensor.clamp(-1, 1)  # 安全限制幅度
            all_audio.append(final_tensor)

        if not all_audio:
            print("[CosyVoice2] 未生成任何音频段。")
            return

        # 拼接所有段：按时间顺序拼接 along time dimension (dim=1)
        combined_audio = torch.cat(all_audio, dim=1)  # (1, total_samples)

        torchaudio.save(out_wav, combined_audio, model.sample_rate)
        print(f"[CosyVoice2] 音频已拼接并保存为: {out_wav}")

    except Exception as e:
        print(f"[错误] CosyVoice2 TTS 拼接或保存失败: {e}")
        import traceback
        traceback.print_exc()

@torch.inference_mode()
def generate_text_to_audio(
    prompt: str,
    negative_prompt: str,
    model_bundle: dict,
    duration: float = 8.0,
    cfg_strength: float = 4.5,
    num_steps: int = 100,
    seed: int = 42,
    output_path: Path = Path('./output/out.wav'),
    normalize: bool = True,
    volume: float = -23.0,
    peak_norm_db_for_norm: float = -1.0,
):
    setup_eval_logging()

    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seq_cfg = model_bundle['seq_cfg']
    net = model_bundle['net']
    feature_utils = model_bundle['feature_utils']
    device = model_bundle['device']

    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=num_steps)

    # 无视频模式
    clip_frames = sync_frames = None
    seq_cfg.duration = duration
    net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

    log.info(f'Prompt: {prompt}')
    log.info(f'Negative prompt: {negative_prompt}')

    audios = generate(
        clip_frames, sync_frames, [prompt],
        negative_text=[negative_prompt],
        feature_utils=feature_utils,
        net=net,
        fm=fm,
        rng=rng,
        cfg_strength=cfg_strength
    )

    audio = audios.float().cpu()[0]  # (channels, samples)

    # ========== 添加响度归一化处理 ==========
    if normalize:
        audio_np = audio.numpy()
        normalized_audio_np = LOUDNESS_NORM(
            audio_np,
            sr=seq_cfg.sampling_rate,
            target_lufs=volume,
            peak_norm_db=peak_norm_db_for_norm
        )
        final_audio = torch.from_numpy(normalized_audio_np).to(torch.float32)
        log.info(f'[Normalization] Audio normalized to {volume} LUFS, peak: {peak_norm_db_for_norm} dBFS')
    else:
        final_audio = audio
        log.info('[Normalization] Skipped')
    # =========================================

    final_audio = final_audio.clamp(-1, 1)  # 保证在合法范围内
    torchaudio.save(output_path, final_audio, seq_cfg.sampling_rate)
    log.info(f'Audio saved to {output_path}')

    if device == 'cuda':
        log.info('Memory usage: %.2f GB', torch.cuda.max_memory_allocated() / (2**30))

    return output_path


# ------------------------------
# 主执行逻辑
# ------------------------------
if __name__ == "__main__":
    print("--- 开始执行脚本 ---")

    cosyvoice_model = load_cosyvoice_model()
    # stable_audio_model, _, sa_sample_rate, sa_sample_size = load_stable_audio_model() # config 不再直接使用

    TARGET_LUFS_SPEECH = -20.0
    TARGET_LUFS_MUSIC = -34.0
    PEAK_NORM_DB = -1.0 # 峰值归一化目标，可以按需调整

    if cosyvoice_model:
        print("\n--- 测试 CosyVoice TTS ---")
        tts(
            cosyvoice_model,
            text="哪吒，你明知道你身上背负着魔丸的诅咒，为何还要一再挑衅别人？你就这么不在乎自己吗？",
            speaker_id="敖丙悲伤", # 确保这个 speaker 存在
            normalize=True,
            volume=TARGET_LUFS_SPEECH,
            peak_norm_db_for_norm=PEAK_NORM_DB
        )
        tts(
            cosyvoice_model,
            text="你好世界，我是一个中文语音合成模型。这段语音将会被进行响度归一化处理。",
            speaker_id="李靖高兴", # 使用一个通用 speaker
            out_wav="output_cosyvoice_hello.wav",
            normalize=True,
            volume=TARGET_LUFS_SPEECH,
            peak_norm_db_for_norm=PEAK_NORM_DB
        )

    if stable_audio_model:
        print("\n--- 测试 Stable Audio ---")
        generate_stable_audio(
            stable_audio_model,
            prompt="epic cinematic trailer music, powerful brass, dramatic strings, 80 BPM",
            duration=8, # 缩短时长以便快速测试
            current_sample_rate=sa_sample_rate,
            current_sample_size=sa_sample_size,
            output_path="output_stable_audio_cinematic.wav",
            normalize=True,
            target_lufs=TARGET_LUFS_MUSIC,
            peak_norm_db_for_norm=PEAK_NORM_DB
        )
        generate_stable_audio(
            stable_audio_model,
            prompt="calm lofi hip hop beat, chill, relaxing, vinyl crackle, short loop",
            duration=10,
            current_sample_rate=sa_sample_rate,
            current_sample_size=sa_sample_size,
            output_path="output_stable_audio_lofi.wav",
            normalize=True,
            target_lufs=TARGET_LUFS_MUSIC,
            peak_norm_db_for_norm=PEAK_NORM_DB
        )

    print("\n--- 脚本执行完毕 ---")
