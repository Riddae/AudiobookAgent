import os
import json5
import utils

def normalize_audio_type(audio_type):
    mapping = {
        'bgm': 'music',
        'sfx': 'sound_effect',
        'speech': 'speech'
    }
    return mapping.get(audio_type, audio_type)

def collect_and_check_audio_data(data):
    fg_audio_id = 0
    fg_audios = []
    bg_audio_dict = {}

    for audio in data:
        # 补全字段兼容
        audio['audio_type'] = normalize_audio_type(audio['audio_type'])
        if 'character' not in audio and 'speaker' in audio:
            audio['character'] = audio['speaker']
        if 'len' not in audio and 'duration' in audio:
            audio['len'] = audio['duration']

        if audio['layout'] == 'foreground':
            audio['id'] = fg_audio_id
            fg_audios.append(audio)
            fg_audio_id += 1
        else:  # background
            if audio['action'] == 'start':
                audio['begin_fg_audio_id'] = audio.get('begin_fg_audio_id', fg_audio_id)
                bg_audio_dict[audio['id']] = audio
            elif audio['action'] == 'stop':
                bg_audio = bg_audio_dict.get(audio['id'])
                if not bg_audio:
                    raise ValueError(f"Stop without start: id={audio['id']}")
                bg_audio['end_fg_audio_id'] = audio.get('end_fg_audio_id', fg_audio_id)

    # 校验完整性
    bg_audios = list(bg_audio_dict.values())
    for bg_audio in bg_audios:
        if 'begin_fg_audio_id' not in bg_audio:
            raise ValueError(f'begin of background missing, audio={bg_audio}')
        if 'end_fg_audio_id' not in bg_audio:
            raise ValueError(f'end of background missing, audio={bg_audio}')
        if bg_audio['begin_fg_audio_id'] > bg_audio['end_fg_audio_id']:
            raise ValueError(f'background audio ends before start, audio={bg_audio}')
        if bg_audio['begin_fg_audio_id'] == bg_audio['end_fg_audio_id']:
            raise ValueError(f'background audio contains no foreground audio, audio={bg_audio}')

    return fg_audios, bg_audios


class AudioCodeGenerator:
    def __init__(self):
        self.wav_counters = {
            'bg_sound_effect': 0,
            'bg_music': 0,
            'idle': 0,
            'fg_sound_effect': 0,
            'fg_music': 0,
            'fg_speech': 0,
        }
        self.code = ''
    
    def append_code(self, content):
        self.code += content + '\n'

    def generate_code(self, fg_audios, bg_audios, output_path, result_filename):
        def get_wav_name(audio):
            audio_type = normalize_audio_type(audio['audio_type'])
            layout = 'fg' if audio['layout'] == 'foreground' else 'bg'
            wav_type = f'{layout}_{audio_type}'
            desc = audio.get('text', audio.get('desc', ''))
            desc = utils.text_to_abbrev_prompt(desc)
            wav_filename = f'{wav_type}_{self.wav_counters[wav_type]}_{desc}.wav'
            self.wav_counters[wav_type] += 1
            return wav_filename

        header = f'''
import os
import sys
import datetime
import time
import torch
from utils import MIX, CAT, COMPUTE_LEN,LOOP
from model import tts
from Audio import audio
fg_audio_lens = []
wav_path = \"{output_path.absolute()}/audio\"
os.makedirs(wav_path, exist_ok=True)

'''
        self.append_code(header)

        fg_audio_wavs = []
        for fg_audio in fg_audios:
            wav_name = get_wav_name(fg_audio)
            audio_type = normalize_audio_type(fg_audio['audio_type'])

            if audio_type == 'sound_effect':
                self.append_code(f'audio(model_bundle=audio_model, prompt=\"{fg_audio["desc"]}\", duration={fg_audio["len"]}, volume={fg_audio["vol"]}, negative_prompt=\" \", output_path=os.path.join(wav_path, \"{wav_name}\"))')
                self.append_code(f'torch.cuda.empty_cache()')
            elif audio_type == 'music':
                self.append_code(f'audio(model_bundle=audio_model, prompt=\"{fg_audio["desc"]}\", duration={fg_audio["len"]}, volume={fg_audio["vol"]}, negative_prompt=\" \", output_path=os.path.join(wav_path, \"{wav_name}\"))')
                self.append_code(f'torch.cuda.empty_cache()')
            elif audio_type == 'speech':
                ref_path = ""
                if "npz_path" in self.char_to_voice_map[fg_audio["character"]]:
                    ref_path = self.char_to_voice_map[fg_audio["character"]]["npz_path"]
                if "wav_path" in self.char_to_voice_map[fg_audio["character"]]:
                    ref_path = self.char_to_voice_map[fg_audio["character"]]["wav_path"]
                ref_full_path = os.path.abspath(ref_path) if os.path.exists(ref_path) else ref_path
                self.append_code(
                    f'tts(model=cosyvoice_model,text=\"{fg_audio["text"]}\", speaker_id=\"{fg_audio["speaker"]}\", volume={fg_audio["vol"]}, out_wav=os.path.join(wav_path, \"{wav_name}\"))'
                )
                self.append_code(f'torch.cuda.empty_cache()')

            fg_audio_wavs.append(wav_name)
            self.append_code(f'fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, \"{wav_name}\")))\n')

        self.append_code('fg_audio_wavs = []')
        for wav in fg_audio_wavs:
            self.append_code(f'fg_audio_wavs.append(os.path.join(wav_path, \"{wav}\"))')
        self.append_code('CAT(wavs=fg_audio_wavs, out_wav=os.path.join(wav_path, \"foreground.wav\"))')

        bg_audio_wavs = []
        self.append_code('\nbg_audio_offsets = []')
        for bg_audio in bg_audios:
            wav_name = get_wav_name(bg_audio)
            audio_type = normalize_audio_type(bg_audio['audio_type'])
            begin_id = bg_audio['begin_fg_audio_id']
            end_id = bg_audio['end_fg_audio_id']
            self.append_code(f'bg_audio_len = sum(fg_audio_lens[{begin_id}:{end_id}])')
            self.append_code(f'bg_audio_offset = sum(fg_audio_lens[:{begin_id}])')
            self.append_code('A_len = 30 if bg_audio_len > 30 else bg_audio_len')
            if audio_type == 'sound_effect':
                self.append_code(f'audio(model_bundle=audio_model,prompt=\"{bg_audio["desc"]}\", volume={bg_audio["vol"]}, duration=A_len, negative_prompt=\" \", output_path=os.path.join(wav_path, \"{wav_name}\"))')
                self.append_code(f'torch.cuda.empty_cache()')
                self.append_code(f'LOOP(os.path.join(wav_path, \"{wav_name}\"), os.path.join(wav_path, \"{wav_name}\"), bg_audio_len)')
            elif audio_type == 'music':
                self.append_code(f'audio(model_bundle=audio_model,prompt=\"{bg_audio["desc"]}\", volume={bg_audio["vol"]}, duration=A_len, negative_prompt=\" \", output_path=os.path.join(wav_path, \"{wav_name}\"))')
                self.append_code(f'torch.cuda.empty_cache()')
                self.append_code(f'LOOP(os.path.join(wav_path, \"{wav_name}\"), os.path.join(wav_path, \"{wav_name}\"), bg_audio_len)')
            else:
                raise ValueError(f"Unsupported background audio_type: {audio_type}")
            bg_audio_wavs.append(wav_name)
            self.append_code('bg_audio_offsets.append(bg_audio_offset)\n')

        self.append_code('bg_audio_wavs = []')
        for wav in bg_audio_wavs:
            self.append_code(f'bg_audio_wavs.append(os.path.join(wav_path, \"{wav}\"))')

        self.append_code('bg_audio_wav_offset_pairs = list(zip(bg_audio_wavs, bg_audio_offsets))')
        self.append_code('bg_audio_wav_offset_pairs.append((os.path.join(wav_path, \"foreground.wav\"), 0))')
        self.append_code(f'MIX(wavs=bg_audio_wav_offset_pairs, out_wav=os.path.join(wav_path, \"{result_filename}.wav\"))')

    def init_char_to_voice_map(self, filename):
        with open(filename, 'r') as file:
            self.char_to_voice_map = json5.load(file)

    def parse_and_generate(self, script_filename, char_to_voice_map_filename, output_path, result_filename='result'):
        self.code = ''
        self.init_char_to_voice_map(char_to_voice_map_filename)
        data = []
        with open(script_filename, 'r') as file:
            for line in file:
                line = line.strip()
                if line:
                    json_object = json5.loads(line)
                    data.append(json_object)
        fg_audios, bg_audios = collect_and_check_audio_data(data)
        self.generate_code(fg_audios, bg_audios, output_path, result_filename)
        return self.code
