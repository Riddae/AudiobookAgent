
import os
import sys
import datetime
import time
import torch
from utils import MIX, CAT, COMPUTE_LEN,LOOP
from model import tts
from Audio import audio
fg_audio_lens = []
wav_path = "/cpfs01/user/renyiming/AudiobookAgent/out_gpt/audio"
os.makedirs(wav_path, exist_ok=True)


audio(model_bundle=audio_model, prompt="Neon lights flickering and crowd ambient chatter", duration=3, volume=-34, negative_prompt=" ", output_path=os.path.join(wav_path, "fg_sound_effect_0_Neon_lights_flickering_and_crowd.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_0_Neon_lights_flickering_and_crowd.wav")))

tts(model=cosyvoice_model,text="夜幕降临，霓虹灯初上，唐僧师徒四人踏进了都市中心最热闹的夜店。音乐震耳欲聋，灯光如水流转，异域气息扑面而来。", speaker_id="旁白", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_0_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_0_.wav")))

tts(model=cosyvoice_model,text="师傅，这地方好热闹啊！您不是说不问世事吗，怎么今天要来这儿凑热闹？", speaker_id="孙悟空", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_1_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_1_.wav")))

audio(model_bundle=audio_model, prompt="People laughing and glasses clinking", duration=2, volume=-36, negative_prompt=" ", output_path=os.path.join(wav_path, "fg_sound_effect_1_People_laughing_and_glasses_clinking.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_1_People_laughing_and_glasses_clinking.wav")))

tts(model=cosyvoice_model,text="悟空，此处人声鼎沸，别有洞天。为师也想亲身体验一下凡尘百态，增长见识。", speaker_id="唐僧", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_2_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_2_.wav")))

tts(model=cosyvoice_model,text="师傅英明啊！师兄，我听说这里的烤串特别香，酒水也都是仙酿，不如咱们先尝尝？", speaker_id="猪八戒", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_3_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_3_.wav")))

audio(model_bundle=audio_model, prompt="Barbecue skewers sizzling on a grill", duration=2, volume=-38, negative_prompt=" ", output_path=os.path.join(wav_path, "fg_sound_effect_2_Barbecue_skewers_sizzling_on_a.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_2_Barbecue_skewers_sizzling_on_a.wav")))

tts(model=cosyvoice_model,text="二师兄，可别贪杯误了正事。师傅，您想坐哪儿？要不我们找个安静点的位置？", speaker_id="李靖", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_4_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_4_.wav")))

tts(model=cosyvoice_model,text="就随你们安排吧，但不可胡来。务必记得，莫要招惹是非。", speaker_id="唐僧", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_5_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_5_.wav")))

audio(model_bundle=audio_model, prompt="Soft footsteps and chairs moving against wooden floor", duration=3, volume=-40, negative_prompt=" ", output_path=os.path.join(wav_path, "fg_sound_effect_3_Soft_footsteps_and_chairs_moving.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_3_Soft_footsteps_and_chairs_moving.wav")))

tts(model=cosyvoice_model,text="放心，师傅！有俺老孙在，没人敢惹咱们。", speaker_id="孙悟空", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_6_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_6_.wav")))

tts(model=cosyvoice_model,text="嘿嘿嘿，那俺就去把烤串点上！沙师弟，看好师傅，别让他喝多了！", speaker_id="猪八戒", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_7_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_7_.wav")))

audio(model_bundle=audio_model, prompt="Order bell ringing at a bar counter", duration=2, volume=-39, negative_prompt=" ", output_path=os.path.join(wav_path, "fg_sound_effect_4_Order_bell_ringing_at_a.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_4_Order_bell_ringing_at_a.wav")))

tts(model=cosyvoice_model,text="好的，二师兄。师傅，您要喝点什么？这里有果汁、茶，还有……呃，仙人特酿。", speaker_id="李靖", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_8_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_8_.wav")))

tts(model=cosyvoice_model,text="为师只要一杯清茶便好。悟空，你可别闹事。", speaker_id="唐僧", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_9_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_9_.wav")))

audio(model_bundle=audio_model, prompt="Dance floor sounds: sneakers sliding and crowd cheering as music intensifies", duration=3, volume=-37, negative_prompt=" ", output_path=os.path.join(wav_path, "fg_sound_effect_5_Dance_floor_sounds_sneakers_sliding.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_5_Dance_floor_sounds_sneakers_sliding.wav")))

tts(model=cosyvoice_model,text="师徒四人就这样在夜店中各自忙碌，唐僧端坐一隅，李靖细心照看。孙悟空已悄然溜到舞池边，一展“筋斗舞技”，猪八戒更是大快朵颐，乐在其中。", speaker_id="旁白", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_10_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_10_.wav")))

fg_audio_wavs = []
fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_0_Neon_lights_flickering_and_crowd.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_0_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_1_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_1_People_laughing_and_glasses_clinking.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_2_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_3_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_2_Barbecue_skewers_sizzling_on_a.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_4_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_5_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_3_Soft_footsteps_and_chairs_moving.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_6_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_7_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_4_Order_bell_ringing_at_a.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_8_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_9_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_5_Dance_floor_sounds_sneakers_sliding.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_10_.wav"))
CAT(wavs=fg_audio_wavs, out_wav=os.path.join(wav_path, "foreground.wav"))

bg_audio_offsets = []
bg_audio_len = sum(fg_audio_lens[0:17])
bg_audio_offset = sum(fg_audio_lens[:0])
A_len = 30 if bg_audio_len > 30 else bg_audio_len
audio(model_bundle=audio_model,prompt="Thumping nightclub electronic music with global flavors, energetic and busy", volume=-37, duration=A_len, negative_prompt=" ", output_path=os.path.join(wav_path, "bg_music_0_Thumping_nightclub_electronic_music_with.wav"))
torch.cuda.empty_cache()
LOOP(os.path.join(wav_path, "bg_music_0_Thumping_nightclub_electronic_music_with.wav"), os.path.join(wav_path, "bg_music_0_Thumping_nightclub_electronic_music_with.wav"), bg_audio_len)
bg_audio_offsets.append(bg_audio_offset)

bg_audio_wavs = []
bg_audio_wavs.append(os.path.join(wav_path, "bg_music_0_Thumping_nightclub_electronic_music_with.wav"))
bg_audio_wav_offset_pairs = list(zip(bg_audio_wavs, bg_audio_offsets))
bg_audio_wav_offset_pairs.append((os.path.join(wav_path, "foreground.wav"), 0))
MIX(wavs=bg_audio_wav_offset_pairs, out_wav=os.path.join(wav_path, "final_mix.wav"))
