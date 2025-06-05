
import os
import sys
import datetime
import time
import torch
from utils import MIX, CAT, COMPUTE_LEN,LOOP
from model import tts
from Audio import audio
fg_audio_lens = []
wav_path = "/cpfs01/user/renyiming/AudiobookAgent/output/audio"
os.makedirs(wav_path, exist_ok=True)


tts(model=cosyvoice_model,text="师徒四人行路疲惫，黄昏时分在一片荒山脚下歇息。唐僧入定打坐，沙僧默默整理着行囊和马匹。猪八戒一屁股重重坐在地上，蒲扇般的大耳朵无力地耷拉着，他呼哧呼哧地喘着粗气，不停用那件打了补丁的破僧衣袖子擦着额头和脸颊上的汗珠。", speaker_id="旁白", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_0_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_0_.wav")))

audio(model_bundle=audio_model, prompt="Deep, heavy, tired panting sounds from a large creature, wheezing slightly, out of breath. Sound effect.", duration=2.0, volume=-20, negative_prompt=" ", output_path=os.path.join(wav_path, "fg_sound_effect_0_Deep_heavy_tired_panting_sounds.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_0_Deep_heavy_tired_panting_sounds.wav")))

tts(model=cosyvoice_model,text="哎哟喂，累死俺老猪了！猴哥，你说咱们这取经的路，它、它啥时候才是个头啊？这走了千山又万水，俺老猪这两条腿哟，都快磨成绣花针了！肚子也饿得咕咕叫唤哩！", speaker_id="猪八戒", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_1_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_1_.wav")))

tts(model=cosyvoice_model,text="正在附近一棵老树上警惕瞭望，顺便寻摸些野果的孙悟空，听见八戒这熟悉的抱怨，身形一纵，一个筋斗便轻巧地翻落到八戒面前。", speaker_id="旁白", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_2_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_2_.wav")))

audio(model_bundle=audio_model, prompt="Quick, sharp whoosh sound of fast movement through air, followed by a light, agile landing thud on soft dirt. Sound effect.", duration=2.0, volume=-20, negative_prompt=" ", output_path=os.path.join(wav_path, "fg_sound_effect_1_Quick_sharp_whoosh_sound_of.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_1_Quick_sharp_whoosh_sound_of.wav")))

tts(model=cosyvoice_model,text=" 嘿！你这呆子！才走了几步路，就又开始哼哼唧唧喊累了？想当年俺老孙一个筋斗云便是十万八千里，眼皮都不带眨一下！你这夯货，就知道吃！就知道睡！除了这两样，你还会点别的吗？", speaker_id="孙悟空", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_3_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_3_.wav")))

tts(model=cosyvoice_model,text="猴哥，话可不能这么说。俺老猪这身板，底子厚实，消耗自然就大嘛！哪像你，瘦得跟个细柴棍儿似的，风一吹就能刮跑。再说，这光走路不给饱饭吃，铁打的身子也顶不住啊！要是能让俺老猪回高老庄，有翠兰给俺端上一大碗香喷喷的白米饭，再来几盘可口的素菜，哎哟，那才叫神仙过的日子哩！", speaker_id="猪八戒", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_4_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_4_.wav")))

tts(model=cosyvoice_model,text="孙悟空听八戒又提到高老庄和他的翠兰，更是气不打一处来，伸出手指几乎要戳到八戒的鼻子。", speaker_id="旁白", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_5_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_5_.wav")))

tts(model=cosyvoice_model,text="呔！你这呆子，凡心又动了是不是！师父的安危还没个着落，西天的真经八字还没一撇，你就惦记着回你的温柔乡，做你的快活女婿？我看你这猪头是几天没挨老孙的棒子，皮又痒痒了，想尝尝俺金箍棒的滋味不成？！", speaker_id="孙悟空", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_6_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_6_.wav")))

tts(model=cosyvoice_model,text="见孙悟空似乎真有些动怒，还作势要摸耳朵里的金箍棒，猪八戒吓得脖子一缩，脸上的肥肉都颤了颤，连忙摆手讨饶。", speaker_id="旁白", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_7_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_7_.wav")))

tts(model=cosyvoice_model,text="诶诶诶，猴哥，猴哥息怒，息怒！千万息怒！俺老猪就是……就是随口那么一说，说说而已，活跃活跃气氛嘛！谁不知道猴哥你本事通天，神通广大，这取经的大业，全得仰仗您老人家呢！俺老猪就是……就是这肚子，它实在饿得慌，前胸都快贴后背了。", speaker_id="猪八戒", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_8_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_8_.wav")))

tts(model=cosyvoice_model,text="孙悟空瞧他那副熊样，虽然依旧板着脸，但眼神稍缓，从怀里摸出几个刚摘的野果，没好气地丢了过去。", speaker_id="旁白", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_9_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_9_.wav")))

audio(model_bundle=audio_model, prompt="Rustle of cloth from a pocket or pouch, followed by a few small, hard round objects (like nuts or small, firm fruits) gently clinking or rolling together and then being tossed lightly. Sound effect. ", duration=2.0, volume=-20, negative_prompt=" ", output_path=os.path.join(wav_path, "fg_sound_effect_2_Rustle_of_cloth_from_a.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_2_Rustle_of_cloth_from_a.wav")))

tts(model=cosyvoice_model,text="哼！就知道吃！拿去，先垫垫你那无底洞似的肚子吧！等沙师弟生好了火，看看能不能化些斋饭来。给俺老孙记住了，少在这儿唉声叹气，拖拖拉拉，要是耽误了取经的大事，看俺老孙不把你这猪耳朵拧下来当下酒菜！", speaker_id="孙悟空", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_10_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_10_.wav")))

tts(model=cosyvoice_model,text="一见到吃的，猪八戒方才的疲惫和委屈顿时烟消云散，他手忙脚乱地接过野果，也顾不上擦拭，便急吼吼地往嘴里塞了一个，腮帮子撑得鼓鼓囊囊。", speaker_id="旁白", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_11_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_11_.wav")))

audio(model_bundle=audio_model, prompt="Satisfied, slightly piggy grunt or happy, muffled chewing sounds. Sound effect.", duration=2.0, volume=-20, negative_prompt=" ", output_path=os.path.join(wav_path, "fg_sound_effect_3_Satisfied_slightly_piggy_grunt_or.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_3_Satisfied_slightly_piggy_grunt_or.wav")))

tts(model=cosyvoice_model,text="嘿嘿，谢谢猴哥，谢谢猴哥！还是猴哥心疼俺老猪！猴哥你放心，只要有吃的填饱肚子，俺老猪保管精神百倍，腿脚利索，一路上绝不拖后腿！", speaker_id="猪八戒", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_12_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_12_.wav")))

tts(model=cosyvoice_model,text="孙悟空没好气地白了他一眼，不再理会，又是一个纵身，轻巧地跃回了树梢，目光锐利地继续扫视着四周的动静。", speaker_id="旁白", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_13_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_13_.wav")))

tts(model=cosyvoice_model,text="但愿如此。你这呆子，少在这儿说大话糊弄俺！", speaker_id="孙悟空", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_14_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_14_.wav")))

tts(model=cosyvoice_model,text="猪八戒一边津津有味地啃着酸甜的野果，一边偷偷瞄着树上孙悟空那警惕的背影，嘴里还满足地小声嘀咕着。", speaker_id="旁白", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_15_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_15_.wav")))

tts(model=cosyvoice_model,text="嘿，这弼马温，就是嘴硬心软……这野果子，还真甜！", speaker_id="猪八戒", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_16_.wav"))
torch.cuda.empty_cache()
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_16_.wav")))

fg_audio_wavs = []
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_0_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_0_Deep_heavy_tired_panting_sounds.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_1_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_2_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_1_Quick_sharp_whoosh_sound_of.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_3_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_4_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_5_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_6_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_7_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_8_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_9_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_2_Rustle_of_cloth_from_a.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_10_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_11_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_3_Satisfied_slightly_piggy_grunt_or.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_12_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_13_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_14_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_15_.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_16_.wav"))
CAT(wavs=fg_audio_wavs, out_wav=os.path.join(wav_path, "foreground.wav"))

bg_audio_offsets = []
bg_audio_len = sum(fg_audio_lens[0:7])
bg_audio_offset = sum(fg_audio_lens[:0])
A_len = 30 if bg_audio_len > 30 else bg_audio_len
audio(model_bundle=audio_model,prompt="Peaceful forest ambiance, gentle breeze rustling leaves, There were some bird calls in the distance, soft flowing creek in the background, tranquil and serene atmosphere, immersive spatial soundscape, high fidelity 44.1k stereo, ambient nature background, perfect for meditation, deep focus or sleep aid, layered organic textures, no human sounds", volume=-38, duration=A_len, negative_prompt=" ", output_path=os.path.join(wav_path, "bg_music_0_Peaceful_forest_ambiance_gentle_breeze.wav"))
torch.cuda.empty_cache()
LOOP(os.path.join(wav_path, "bg_music_0_Peaceful_forest_ambiance_gentle_breeze.wav"), os.path.join(wav_path, "bg_music_0_Peaceful_forest_ambiance_gentle_breeze.wav"), bg_audio_len)
bg_audio_offsets.append(bg_audio_offset)

bg_audio_len = sum(fg_audio_lens[7:21])
bg_audio_offset = sum(fg_audio_lens[:7])
A_len = 30 if bg_audio_len > 30 else bg_audio_len
audio(model_bundle=audio_model,prompt="Short, playful, slightly comical traditional Chinese folk motif, Pipa and Dizi, upbeat. Instrumental.", volume=-38, duration=A_len, negative_prompt=" ", output_path=os.path.join(wav_path, "bg_music_1_Short_playful_slightly_comical_traditional.wav"))
torch.cuda.empty_cache()
LOOP(os.path.join(wav_path, "bg_music_1_Short_playful_slightly_comical_traditional.wav"), os.path.join(wav_path, "bg_music_1_Short_playful_slightly_comical_traditional.wav"), bg_audio_len)
bg_audio_offsets.append(bg_audio_offset)

bg_audio_wavs = []
bg_audio_wavs.append(os.path.join(wav_path, "bg_music_0_Peaceful_forest_ambiance_gentle_breeze.wav"))
bg_audio_wavs.append(os.path.join(wav_path, "bg_music_1_Short_playful_slightly_comical_traditional.wav"))
bg_audio_wav_offset_pairs = list(zip(bg_audio_wavs, bg_audio_offsets))
bg_audio_wav_offset_pairs.append((os.path.join(wav_path, "foreground.wav"), 0))
MIX(wavs=bg_audio_wav_offset_pairs, out_wav=os.path.join(wav_path, "final_mix.wav"))
