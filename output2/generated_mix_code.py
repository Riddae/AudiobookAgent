
import os
import sys
import datetime
import time
import torch
from utils import MIX, CAT, COMPUTE_LEN,LOOP
from model import tts
from Audio import audio
fg_audio_lens = []
wav_path = "/cpfs01/user/renyiming/AudiobookAgent/output2/audio"
os.makedirs(wav_path, exist_ok=True)


audio(model_bundle=audio_model, prompt="Sudden strong wind gust, rocks crumbling, and a magical whoosh sound", duration=8, volume=-10, negative_prompt=" ", output_path=os.path.join(wav_path, "fg_sound_effect_0_Sudden_strong_wind_gust_rocks.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_0_Sudden_strong_wind_gust_rocks.wav")))

tts(model=cosyvoice_model,text="唐僧师徒四人行至一处山谷，忽然之间，风云变幻，山石崩裂，仿佛被一股无形之力席卷。待得尘埃落定，眼前已不再是荒山野岭，而是灯火璀璨、车水马龙的奇异景象。", speaker_id="旁白", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_0_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_0_.wav")))

audio(model_bundle=audio_model, prompt="Loud city traffic with car horns and passing vehicles", duration=8, volume=-20, negative_prompt=" ", output_path=os.path.join(wav_path, "fg_sound_effect_1_Loud_city_traffic_with_car.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_1_Loud_city_traffic_with_car.wav")))

tts(model=cosyvoice_model,text="阿弥陀佛！这、这是何处？吾等方才还在荒野，怎地眨眼间便入了这等奇诡之地？", speaker_id="唐僧", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_1_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_1_.wav")))

tts(model=cosyvoice_model,text="呔！哪里来的妖雾？竟能瞒过俺老孙的火眼金睛！师父莫慌，待我看看这究竟是何方妖孽布下的迷阵！", speaker_id="孙悟空", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_2_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_2_.wav")))

audio(model_bundle=audio_model, prompt="Protracted stomach rumbling sound", duration=2, volume=-28, negative_prompt=" ", output_path=os.path.join(wav_path, "fg_sound_effect_2_Protracted_stomach_rumbling_sound.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_2_Protracted_stomach_rumbling_sound.wav")))

tts(model=cosyvoice_model,text="哎哟喂，我的老腰！师父，是不是到斋饭的地方了？徒儿肚子饿得咕咕叫了。哇！好多亮光！是不是什么仙境啊？看着可比那山洞里舒坦多了！", speaker_id="猪八戒", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_3_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_3_.wav")))

tts(model=cosyvoice_model,text="大师兄，这不像凡间。你看，那些铁皮盒子竟然自己跑动，速度奇快，还发出轰鸣之声。还有这些高耸入云的建筑，不似寻常寺庙山峰。", speaker_id="沙悟净", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_4_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_4_.wav")))

tts(model=cosyvoice_model,text="师徒四人站在高楼林立、霓虹闪烁的街道中央，显得格格不入。疾驰而过的汽车、刺耳的喇叭声、以及来往穿梭的人群，都让他们感到前所未有的陌生。", speaker_id="旁白", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_5_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_5_.wav")))

audio(model_bundle=audio_model, prompt="General city ambience with distant traffic and indistinct chatter", duration=8, volume=-25, negative_prompt=" ", output_path=os.path.join(wav_path, "fg_sound_effect_3_General_city_ambience_with_distant.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_3_General_city_ambience_with_distant.wav")))

tts(model=cosyvoice_model,text="哼！这些“铁皮怪兽”跑得比马还快，也不见有人驭使！还有这些“钢筋水泥山”，直插云霄，比那花果山还高上几分！莫非是天上仙人所居之处？", speaker_id="孙悟空", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_6_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_6_.wav")))

audio(model_bundle=audio_model, prompt="Protracted stomach rumbling sound", duration=2, volume=-28, negative_prompt=" ", output_path=os.path.join(wav_path, "fg_sound_effect_4_Protracted_stomach_rumbling_sound.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_4_Protracted_stomach_rumbling_sound.wav")))

tts(model=cosyvoice_model,text="哎呀！师父，你看那边！亮晃晃的，还有好多好吃的！是不是饭店啊？看着比高老庄的酒肆还气派！俺老猪可走不动了，咱们进去歇歇脚，顺便化些斋饭吧！", speaker_id="猪八戒", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_7_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_7_.wav")))

tts(model=cosyvoice_model,text="悟空，八戒，悟净，莫要轻举妄动。此地太过诡异，为师总觉心中不安。这些凡人衣着奇特，行色匆匆，难道皆是受了妖魔的迷惑，亦或是身处幻境之中？", speaker_id="唐僧", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_8_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_8_.wav")))

tts(model=cosyvoice_model,text="师父说的是。此地不见山林野兽，亦无田间阡陌，更不见寻常百姓劳作。处处光怪陆离，与我等所经历的凡世截然不同。", speaker_id="沙悟净", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_9_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_9_.wav")))

tts(model=cosyvoice_model,text="管他什么妖魔鬼怪，俺老孙去探个究竟！看他们往来的这些凡人，倒也无甚妖气。兴许是哪里来的新地界呢！", speaker_id="孙悟空", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_10_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_10_.wav")))

tts(model=cosyvoice_model,text="悟空，切莫冲动。你且去探探虚实，但要小心谨慎，莫要惊扰了凡人。八戒、悟净，你二人护好行李与白马，在此等候。", speaker_id="唐僧", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_11_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_11_.wav")))

tts(model=cosyvoice_model,text="哎，大师兄，你可别只顾着探路，顺便去那亮堂堂的地方，问问有没有上好的素斋！俺老猪饿得前胸贴后背了！", speaker_id="猪八戒", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_12_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_12_.wav")))

tts(model=cosyvoice_model,text="二师兄，此地古怪，一切尚不明朗。我等当以师父安危和取经大业为重。", speaker_id="沙悟净", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_13_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_13_.wav")))

tts(model=cosyvoice_model,text="面对这突如其来的现代都市，师徒四人各怀心事。他们的西行之路，似乎在这一刻，拐向了意想不到的岔口。", speaker_id="旁白", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_14_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_14_.wav")))

fg_audio_wavs = []
fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_0_Sudden_strong_wind_gust_rocks.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_0_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_1_Loud_city_traffic_with_car.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_1_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_2_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_2_Protracted_stomach_rumbling_sound.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_3_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_4_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_5_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_3_General_city_ambience_with_distant.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_6_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_4_Protracted_stomach_rumbling_sound.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_7_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_8_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_9_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_10_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_11_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_12_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_13_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_14_.wav"))
CAT(wavs=fg_audio_wavs, out_wav=os.path.join(wav_path, "foreground.wav"))

bg_audio_offsets = []
bg_audio_len = sum(fg_audio_lens[0:2])
bg_audio_offset = sum(fg_audio_lens[:0])
A_len = 30 if bg_audio_len > 30 else bg_audio_len
audio(model_bundle=audio_model,prompt="Mysterious and unsettling atmospheric music", volume=-40, duration=A_len, negative_prompt=" ", output_path=os.path.join(wav_path, "bg_music_0_Mysterious_and_unsettling_atmospheric_music.wav"))
torch.cuda.empty_cache()
LOOP(os.path.join(wav_path, "bg_music_0_Mysterious_and_unsettling_atmospheric_music.wav"), os.path.join(wav_path, "bg_music_0_Mysterious_and_unsettling_atmospheric_music.wav"), bg_audio_len)
bg_audio_offsets.append(bg_audio_offset)

bg_audio_len = sum(fg_audio_lens[2:20])
bg_audio_offset = sum(fg_audio_lens[:2])
A_len = 30 if bg_audio_len > 30 else bg_audio_len
audio(model_bundle=audio_model,prompt="Busy modern city ambience with distant traffic and indistinct chatter", volume=-38, duration=A_len, negative_prompt=" ", output_path=os.path.join(wav_path, "bg_music_1_Busy_modern_city_ambience_with.wav"))
torch.cuda.empty_cache()
LOOP(os.path.join(wav_path, "bg_music_1_Busy_modern_city_ambience_with.wav"), os.path.join(wav_path, "bg_music_1_Busy_modern_city_ambience_with.wav"), bg_audio_len)
bg_audio_offsets.append(bg_audio_offset)

bg_audio_wavs = []
bg_audio_wavs.append(os.path.join(wav_path, "bg_music_0_Mysterious_and_unsettling_atmospheric_music.wav"))
bg_audio_wavs.append(os.path.join(wav_path, "bg_music_1_Busy_modern_city_ambience_with.wav"))
bg_audio_wav_offset_pairs = list(zip(bg_audio_wavs, bg_audio_offsets))
bg_audio_wav_offset_pairs.append((os.path.join(wav_path, "foreground.wav"), 0))
MIX(wavs=bg_audio_wav_offset_pairs, out_wav=os.path.join(wav_path, "final_mix.wav"))
