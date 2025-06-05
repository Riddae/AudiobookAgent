
import os
import sys
import datetime
import time
import torch
from utils import MIX, CAT, COMPUTE_LEN,LOOP
from model import tts
from Audio import audio
fg_audio_lens = []
wav_path = "/cpfs01/user/renyiming/AudiobookAgent/output1/audio"
os.makedirs(wav_path, exist_ok=True)


audio(model_bundle=audio_model, prompt="Sound of desolate wind blowing across an open landscape, with a growing, distant rumble and wash of a massive body of water.", duration=3, volume=-33, negative_prompt=" ", output_path=os.path.join(wav_path, "fg_sound_effect_0_Sound_of_desolate_wind_blowing.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_0_Sound_of_desolate_wind_blowing.wav")))

tts(model=cosyvoice_model,text="师徒四人一路西行，风餐露宿，历尽艰辛。这一日，他们来到一处水天相接的所在，只见白浪滔天，烟波浩渺，挡住了西去的道路。", speaker_id="旁白", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_0_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_0_.wav")))

audio(model_bundle=audio_model, prompt="The full, powerful sound of a vast river with large waves crashing against a shore and strong, gusty wind, conveying immense scale and power.", duration=4, volume=-27, negative_prompt=" ", output_path=os.path.join(wav_path, "fg_sound_effect_1_The_full_powerful_sound_of.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_1_The_full_powerful_sound_of.wav")))

tts(model=cosyvoice_model,text="阿弥陀佛！悟空，你看这水势，比那流沙河、黑水河又不知宽阔多少倍，如何是好啊？", speaker_id="唐僧", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_1_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_1_.wav")))

tts(model=cosyvoice_model,text="我的娘欸！这哪里是河，分明就是个海！师父，这下可真完了！我看呐，咱们还是趁早散伙，我回我的高老庄，猴哥回他的花果山，沙师弟回他的流沙河，您老人家也回东土大唐享福去吧！", speaker_id="猪八戒", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_2_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_2_.wav")))

tts(model=cosyvoice_model,text="呆子！又说这等没出息的丧气话！师父莫愁，待老孙先到水边看看，再驾筋斗云到空中探个究竟，看这河到底有多宽，有无船只可以渡过。", speaker_id="孙悟空", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_3_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_3_.wav")))

tts(model=cosyvoice_model,text="大师兄说的是。师父，凡事都有个解决的办法。大师兄神通广大，定能寻到过河之法。", speaker_id="李靖", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_4_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_4_.wav")))

tts(model=cosyvoice_model,text="唉，也只好如此了。悟空，你此去务必小心，这水域如此广阔，恐有水怪妖邪作祟。", speaker_id="唐僧", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_5_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_5_.wav")))

tts(model=cosyvoice_model,text="师父放心！老孙去去就来！八戒，看好师父和行李，若有差池，俺老孙回来定不饶你！", speaker_id="孙悟空", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_6_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_6_.wav")))

tts(model=cosyvoice_model,text="知道啦，知道啦！猴哥你快去快回，俺老猪也怕水。", speaker_id="猪八戒", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_7_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_7_.wav")))

audio(model_bundle=audio_model, prompt="Sharp, quick 'whoosh' sound effect of a character jumping or moving swiftly to a new position.", duration=1.0, volume=-26, negative_prompt=" ", output_path=os.path.join(wav_path, "fg_sound_effect_2_Sharp_quick_whoosh_sound_effect.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_2_Sharp_quick_whoosh_sound_effect.wav")))

audio(model_bundle=audio_model, prompt="Close-up environmental sound of a very wide and powerful river: strong, deep current sounds, large waves breaking rhythmically, and persistent gusty wind by the water's edge. Imposing and continuous.", duration=5, volume=-27, negative_prompt=" ", output_path=os.path.join(wav_path, "fg_sound_effect_3_Closeup_environmental_sound_of_a.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_3_Closeup_environmental_sound_of_a.wav")))

tts(model=cosyvoice_model,text="孙悟空应了一声，纵身一跃，来到水边。只见这通天河，约摸有八百里宽阔，水流湍急，波涛汹涌，望不到对岸。他又一个筋斗翻上云头，睁开火眼金睛，四下里一望。", speaker_id="旁白", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_8_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_8_.wav")))

audio(model_bundle=audio_model, prompt="Magical 'swoosh' or 'rising wind' sound effect, clear and distinct, indicating someone taking flight rapidly on a cloud or by magical means.", duration=2.5, volume=-25, negative_prompt=" ", output_path=os.path.join(wav_path, "fg_sound_effect_4_Magical_swoosh_or_rising_wind.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_4_Magical_swoosh_or_rising_wind.wav")))

audio(model_bundle=audio_model, prompt="Subtle, brief magical 'shimmer,' 'chime,' or 'focusing' sound effect, indicating the activation of special sight or perception.", duration=1.5, volume=-30, negative_prompt=" ", output_path=os.path.join(wav_path, "fg_sound_effect_5_Subtle_brief_magical_shimmer_chime.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_5_Subtle_brief_magical_shimmer_chime.wav")))

audio(model_bundle=audio_model, prompt="Low, ominous, and unsettling drone or deep humming sound, perhaps with a slight watery distortion or echo, suggesting the discovery of a hidden, malevodurationt presence.", duration=3.0, volume=-29, negative_prompt=" ", output_path=os.path.join(wav_path, "fg_sound_effect_6_Low_ominous_and_unsettling_drone.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_6_Low_ominous_and_unsettling_drone.wav")))

tts(model=cosyvoice_model,text="嘿！这河面果然宽广！嗯？奇怪，这水底深处，似乎有一股若有若无的妖气……且待我仔细看看对岸有无人烟。", speaker_id="孙悟空", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_9_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_9_.wav")))

tts(model=cosyvoice_model,text="悟空凝神细看，只见那遥远的对岸，隐隐约约似乎有灯火人家。他心中一喜，正要下去回报，忽然，河面上卷起一股腥风，水浪翻滚得更加厉害了。", speaker_id="旁白", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_10_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_10_.wav")))

audio(model_bundle=audio_model, prompt="Sudden, sharp gust of strong, eerie wind with an unnatural, 'fishy' or 'damp cold' quality, accompanied by an abrupt increase in the sound of turbudurationt, viodurationtly crashing water waves and water splashes.", duration=4.5, volume=-22, negative_prompt=" ", output_path=os.path.join(wav_path, "fg_sound_effect_7_Sudden_sharp_gust_of_strong.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_7_Sudden_sharp_gust_of_strong.wav")))

tts(model=cosyvoice_model,text="八戒，悟净，你们看，河面上似乎起了变化！莫不是悟空遇到了什么麻烦？", speaker_id="唐僧", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_11_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_11_.wav")))

tts(model=cosyvoice_model,text="师父莫慌，大师兄本领高强，应该无碍。只是这风浪确实来得蹊跷。", speaker_id="李靖", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_12_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_12_.wav")))

tts(model=cosyvoice_model,text="哎呀，不会真有什么妖怪出来了吧？猴哥，你可快点回来啊！俺老猪这心里七上八下的！", speaker_id="猪八戒", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_13_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_13_.wav")))

fg_audio_wavs = []
fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_0_Sound_of_desolate_wind_blowing.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_0_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_1_The_full_powerful_sound_of.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_1_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_2_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_3_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_4_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_5_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_6_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_7_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_2_Sharp_quick_whoosh_sound_effect.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_3_Closeup_environmental_sound_of_a.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_8_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_4_Magical_swoosh_or_rising_wind.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_5_Subtle_brief_magical_shimmer_chime.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_6_Low_ominous_and_unsettling_drone.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_9_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_10_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_7_Sudden_sharp_gust_of_strong.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_11_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_12_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_13_.wav"))
CAT(wavs=fg_audio_wavs, out_wav=os.path.join(wav_path, "foreground.wav"))

bg_audio_offsets = []
bg_audio_len = sum(fg_audio_lens[0:11])
bg_audio_offset = sum(fg_audio_lens[:0])
A_len = 30 if bg_audio_len > 30 else bg_audio_len
audio(model_bundle=audio_model,prompt="Ambient music for a long journey, transitioning to an expansive, slightly ominous tone suggesting a vast, unknown river and the chaldurationges ahead.", volume=-38, duration=A_len, negative_prompt=" ", output_path=os.path.join(wav_path, "bg_music_0_Ambient_music_for_a_long.wav"))
torch.cuda.empty_cache()
LOOP(os.path.join(wav_path, "bg_music_0_Ambient_music_for_a_long.wav"), os.path.join(wav_path, "bg_music_0_Ambient_music_for_a_long.wav"), bg_audio_len)
bg_audio_offsets.append(bg_audio_offset)

bg_audio_len = sum(fg_audio_lens[10:22])
bg_audio_offset = sum(fg_audio_lens[:10])
A_len = 30 if bg_audio_len > 30 else bg_audio_len
audio(model_bundle=audio_model,prompt="Investigative and slightly magical underscore music, building gentle tension, with subtle continuous elements of flowing water or deep river rumbles.", volume=-36, duration=A_len, negative_prompt=" ", output_path=os.path.join(wav_path, "bg_music_1_Investigative_and_slightly_magical_underscore.wav"))
torch.cuda.empty_cache()
LOOP(os.path.join(wav_path, "bg_music_1_Investigative_and_slightly_magical_underscore.wav"), os.path.join(wav_path, "bg_music_1_Investigative_and_slightly_magical_underscore.wav"), bg_audio_len)
bg_audio_offsets.append(bg_audio_offset)

bg_audio_wavs = []
bg_audio_wavs.append(os.path.join(wav_path, "bg_music_0_Ambient_music_for_a_long.wav"))
bg_audio_wavs.append(os.path.join(wav_path, "bg_music_1_Investigative_and_slightly_magical_underscore.wav"))
bg_audio_wav_offset_pairs = list(zip(bg_audio_wavs, bg_audio_offsets))
bg_audio_wav_offset_pairs.append((os.path.join(wav_path, "foreground.wav"), 0))
MIX(wavs=bg_audio_wav_offset_pairs, out_wav=os.path.join(wav_path, "final_mix.wav"))
