
import os
import sys
import datetime
import time
import torch
from utils import MIX, CAT, COMPUTE_LEN,LOOP
from model import tts
from Audio import audio
fg_audio_lens = []
wav_path = "/cpfs01/user/renyiming/AudiobookAgent/out3/audio"
os.makedirs(wav_path, exist_ok=True)


audio(model_bundle=audio_model, prompt="Distant ocean waves sound, gradually increasing", duration=3, volume=-30, negative_prompt=" ", output_path=os.path.join(wav_path, "fg_sound_effect_0_Distant_ocean_waves_sound_gradually.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_0_Distant_ocean_waves_sound_gradually.wav")))

audio(model_bundle=audio_model, prompt="Sound of wind blowing through coastal vegetation", duration=2, volume=-25, negative_prompt=" ", output_path=os.path.join(wav_path, "fg_sound_effect_1_Sound_of_wind_blowing_through.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_1_Sound_of_wind_blowing_through.wav")))

audio(model_bundle=audio_model, prompt="Sound of rustling leaves and branches as they push through", duration=2, volume=-28, negative_prompt=" ", output_path=os.path.join(wav_path, "fg_sound_effect_2_Sound_of_rustling_leaves_and.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_2_Sound_of_rustling_leaves_and.wav")))

tts(model=cosyvoice_model,text="唐僧师徒四人一路风尘仆仆，日夜兼程，直奔西天。终于，在又一次翻越峻岭之后，一阵阵隐约的涛声传入耳畔，伴随着咸湿的海风扑面而来。当他们拨开最后的树丛，眼前豁然开朗，一片广袤无垠的蔚蓝大海呈现在眼前。", speaker_id="旁白", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_0_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_0_.wav")))

audio(model_bundle=audio_model, prompt="Sound of gentle ocean waves washing ashore", duration=3, volume=-30, negative_prompt=" ", output_path=os.path.join(wav_path, "fg_sound_effect_3_Sound_of_gentle_ocean_waves.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_3_Sound_of_gentle_ocean_waves.wav")))

tts(model=cosyvoice_model,text="阿弥陀佛……善哉善哉！此处便是海边了？", speaker_id="唐僧", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_1_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_1_.wav")))

tts(model=cosyvoice_model,text="哎呀，师父！这可走到头了！前面竟是水，水，水！放眼望去，连个尽头都没有，这可如何是好？莫不是要我们长出翅膀飞过去不成？", speaker_id="猪八戒", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_2_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_2_.wav")))

audio(model_bundle=audio_model, prompt="Distant seagull cries", duration=2, volume=-35, negative_prompt=" ", output_path=os.path.join(wav_path, "fg_sound_effect_4_Distant_seagull_cries.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_4_Distant_seagull_cries.wav")))

tts(model=cosyvoice_model,text="呆子，莫要聒噪！这便是东海之滨了。师父，徒儿瞧这海域辽阔，波涛汹涌，不见舟楫，恐难渡也。看来寻常法子是过不去了。", speaker_id="孙悟空", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_3_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_3_.wav")))

tts(model=cosyvoice_model,text="大师兄所言不虚。水面浩瀚无边，若无大型船只，恐难轻易渡过。不知前方可有渡口，或能借宿的渔村？", speaker_id="沙悟净", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_4_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_4_.wav")))

tts(model=cosyvoice_model,text="唉，此去西天取经，果真道阻且长。既是海边，前方定有渡口或渔村。悟空，你且去前方探看一番，看是否有船只可渡我们师徒，或是是否有其他蹊径。", speaker_id="唐僧", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_5_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_5_.wav")))

tts(model=cosyvoice_model,text="师父，那可要走多久才到啊？我的肚子早就咕咕叫了，这海边可有什么好吃的？海鲜？螃蟹？要是能捞点上来充饥就好了。", speaker_id="猪八戒", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_6_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_6_.wav")))

tts(model=cosyvoice_model,text="你这呆子，只知吃食！且在此等候，老孙去去就回！", speaker_id="孙悟空", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_7_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_7_.wav")))

audio(model_bundle=audio_model, prompt="Whoosh sound of rapid departure (e.g., a powerful flight)", duration=2, volume=-20, negative_prompt=" ", output_path=os.path.join(wav_path, "fg_sound_effect_5_Whoosh_sound_of_rapid_departure.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_5_Whoosh_sound_of_rapid_departure.wav")))

tts(model=cosyvoice_model,text="说罢，孙悟空一个筋斗云，便化作一道金光，眨眼间消失在天际。唐僧、猪八戒和沙悟净则在海边寻了一处背风的地方，看着眼前浩瀚无垠的蓝色大海，听着海浪拍打礁石的声音，等待着孙悟空的归来。", speaker_id="旁白", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_8_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_8_.wav")))

audio(model_bundle=audio_model, prompt="Continuous sound of ocean waves and distant seagulls", duration=3, volume=-30, negative_prompt=" ", output_path=os.path.join(wav_path, "fg_sound_effect_6_Continuous_sound_of_ocean_waves.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_6_Continuous_sound_of_ocean_waves.wav")))

fg_audio_wavs = []
fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_0_Distant_ocean_waves_sound_gradually.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_1_Sound_of_wind_blowing_through.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_2_Sound_of_rustling_leaves_and.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_0_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_3_Sound_of_gentle_ocean_waves.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_1_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_2_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_4_Distant_seagull_cries.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_3_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_4_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_5_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_6_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_7_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_5_Whoosh_sound_of_rapid_departure.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_8_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_6_Continuous_sound_of_ocean_waves.wav"))
CAT(wavs=fg_audio_wavs, out_wav=os.path.join(wav_path, "foreground.wav"))

bg_audio_offsets = []
bg_audio_len = sum(fg_audio_lens[0:16])
bg_audio_offset = sum(fg_audio_lens[:0])
A_len = 30 if bg_audio_len > 30 else bg_audio_len
audio(model_bundle=audio_model,prompt="Orchestral journey music with a sense of anticipation", volume=-38, duration=A_len, negative_prompt=" ", output_path=os.path.join(wav_path, "bg_music_0_Orchestral_journey_music_with_a.wav"))
torch.cuda.empty_cache()
LOOP(os.path.join(wav_path, "bg_music_0_Orchestral_journey_music_with_a.wav"), os.path.join(wav_path, "bg_music_0_Orchestral_journey_music_with_a.wav"), bg_audio_len)
bg_audio_offsets.append(bg_audio_offset)

bg_audio_wavs = []
bg_audio_wavs.append(os.path.join(wav_path, "bg_music_0_Orchestral_journey_music_with_a.wav"))
bg_audio_wav_offset_pairs = list(zip(bg_audio_wavs, bg_audio_offsets))
bg_audio_wav_offset_pairs.append((os.path.join(wav_path, "foreground.wav"), 0))
MIX(wavs=bg_audio_wav_offset_pairs, out_wav=os.path.join(wav_path, "final_mix.wav"))
