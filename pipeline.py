from pathlib import Path
from code_generation import AudioCodeGenerator
from model import load_cosyvoice_model
from Audio import load_mmaudio_model
from openai import OpenAI
import json
import json5
import argparse
import os
import re

cosyvoice_model = load_cosyvoice_model()
audio_model = load_mmaudio_model(variant='large_44k_v2', full_precision=False)

def get_file_content(filename):
    with open(filename, 'r') as file:
        return file.read().strip()

def extract_substring_with_quotes(input_string, quotes="'''"):
    pattern = f"{quotes}(.*?){quotes}"
    matches = re.findall(pattern, input_string, re.DOTALL)
    for i in range(len(matches)):
        if matches[i][:4] == 'json':
            matches[i] = matches[i][4:]
    
    if len(matches) == 1:
        return matches[0]
    else:
        return matches

def try_extract_content_from_quotes(content):
    if "'''" in content:
        return extract_substring_with_quotes(content)
    elif "```" in content:
        return extract_substring_with_quotes(content, quotes="```")
    else:
        return content

def chat_with_gpt(input):
    client = OpenAI()

    response = client.responses.create(
        model="gpt-4.1",
        input=input
    )
    return response.output_text

def generate_Step1(topic, output_path):
    print("【Step1】Generating guest info ...")
    
    # 1. 构建完整 prompt（读取模板 + 换行 + topic）
    complete_prompt_path = f'prompts/Step1.prompt'
    complete_prompt = get_file_content(complete_prompt_path) + "\n" + topic

    # 2. 获取 GPT 响应
    json_response = try_extract_content_from_quotes(chat_with_gpt(complete_prompt))

    # 3. 保存原始 GPT 响应
    raw_save_path = f'{output_path}/Step1_output.json'
    with open(raw_save_path, 'w', encoding='utf-8') as raw_file:
        raw_file.write(json_response)
    return json_response

def generate_Step2(text, output_path ):
    print("【Step2】Generating guest info ...")
    
    # 1. 构建完整 prompt（读取模板 + 换行 + topic）
    complete_prompt_path = f'prompts/Step2.prompt'
    complete_prompt = get_file_content(complete_prompt_path) + "\n" + text

    # 2. 获取 GPT 响应
    json_response = try_extract_content_from_quotes(chat_with_gpt(complete_prompt))
    print(json_response)
    # 3. 保存原始 GPT 响应
    return json_response

def generate_and_run_audio_script(
    script_path: str,
    char_map_path: str,
    output_dir: str = "output1",
    result_filename: str = "final_mix"
):
    # 初始化生成器
    generator = AudioCodeGenerator()

    # 创建输出目录
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(exist_ok=True)

    # 生成代码字符串
    code = generator.parse_and_generate(
        script_filename=Path(script_path),
        char_to_voice_map_filename=Path(char_map_path),
        output_path=output_dir_path,
        result_filename=result_filename
    )

    # 保存生成代码
    generated_code_path = output_dir_path / "generated_mix_code.py"
    with generated_code_path.open("w", encoding="utf-8") as f:
        f.write(code)

    print("🎉 配音脚本生成完成！请查看：", generated_code_path)

    # 执行生成的配音脚本
    print("✅ 开始运行生成的配音脚本...")
    with generated_code_path.open("r", encoding="utf-8") as f:
        generated_code = f.read()
    exec(generated_code, {
        'cosyvoice_model': cosyvoice_model,
        'audio_model': audio_model,
    })

def process_audio_data(data_list):
    import copy
    temp_items_with_new_id = [copy.deepcopy(item) for item in data_list]
    fg_sequential_id_counter = 0
    # --- Pass 1: Assign sequential id to foreground items ---
    for item in temp_items_with_new_id:
        if item.get("layout") == "foreground":
            item["id"] = fg_sequential_id_counter # Overwrite or add 'id' field
            fg_sequential_id_counter += 1
    # --- Pass 2: Process BGM and link foreground IDs ---
    # `processed_items` will be our final list, built from `temp_items_with_new_id`
    processed_items = []
    open_bgms = {} # Key: bgm_id (from BGM item), Value: {'begin_item_index_in_processed': idx}
    for item_data in temp_items_with_new_id:
        # `item_data` already has the new sequential 'id' for foreground items
        item_type = item_data.get("audio_type")
        item_layout = item_data.get("layout")
        
        # If it's a foreground item, update any open BGMs
        if item_layout == "foreground":
            current_fg_id = item_data["id"] # This is the new sequential ID
            for bgm_tracking_id in open_bgms:
                bgm_info = open_bgms[bgm_tracking_id]
                # Access the BGM "begin" item in the `processed_items` list
                bgm_begin_item_in_final_list = processed_items[bgm_info['begin_item_index_in_processed']]

                if bgm_begin_item_in_final_list.get("begin_fg_audio_id") is None:
                    bgm_begin_item_in_final_list["begin_fg_audio_id"] = current_fg_id
                bgm_begin_item_in_final_list["end_fg_audio_id"] = current_fg_id # Always update last seen

        elif item_type == "bgm":
            bgm_action = item_data.get("action")
            # BGM items use their own "id" for pairing, not the sequential one.
            bgm_pairing_id = item_data.get("id") 

            if bgm_action == "start":
                item_data["begin_fg_audio_id"] = None # Initialize
                item_data["end_fg_audio_id"] = None   # Initialize
                open_bgms[bgm_pairing_id] = {
                    'begin_item_index_in_processed': len(processed_items), # Its future index
                }
            elif bgm_action == "stop":
                if bgm_pairing_id in open_bgms:
                    del open_bgms[bgm_pairing_id]
                else:
                    print(f"Warning: Encountered BGM end for id {bgm_pairing_id} without a corresponding begin.")
        
        processed_items.append(item_data) # Add the (potentially modified) item
        
    return processed_items


def write_to_json5l(data_list, output_filepath):
    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            for item in data_list:
                parts = []
                # Sort keys for consistent output, optional but good for diffs
                # sorted_keys = sorted(item.keys())
                # for key in sorted_keys:
                for key, value in item.items(): # Or iterate directly if order doesn't matter
                    key_str = key
                    if isinstance(value, str):
                        # Use json.dumps for robust string escaping for JSON5 compatibility
                        value_str = json.dumps(value, ensure_ascii=False)
                    elif isinstance(value, bool):
                        value_str = str(value).lower()
                    elif isinstance(value, (int, float)):
                        value_str = str(value)
                    elif value is None: 
                        value_str = 'null'
                    else: # For lists or nested dicts
                        value_str = json.dumps(value, ensure_ascii=False)
                    parts.append(f"{key_str}: {value_str}")
                line = "{" + ", ".join(parts) + "}"
                f.write(line + '\n')
        print(f"Data successfully written to {output_filepath} in JSON5L-like format.")
    except IOError as e:
        print(f"Error writing to file {output_filepath}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during writing: {type(e).__name__} - {e}")

if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="音频自动化生成流程")
    parser.add_argument("--text", type=str, required=True, help="输入的文本主题")
    parser.add_argument("--output_path", type=str, default="output1", help="输出目录")
    args = parser.parse_args()

    # # 创建输出目录
    # os.makedirs(args.output_path, exist_ok=True)

    # # Step 1: 生成嘉宾信息（原始 GPT 响应）
    # step1_json_response = generate_Step1(args.text, args.output_path)

    # # Step 2: 生成逐条配音结构脚本（结构化 JSON）
    # step2_json_response = generate_Step2(step1_json_response, args.output_path)

    # # Step 3: 处理音频结构，生成 ID 并绑定 BGM 区间
    # try:
    #     structured_data = json5.loads(step2_json_response)
    # except Exception as e:
    #     print(f"🛑 无法解析 Step2 响应为 JSON5: {e}")
    #     exit(1)

    # processed_data = process_audio_data(structured_data)

    # # Step 4: 写入为 JSON5L 文件
    json5l_path = os.path.join(args.output_path, "Step2_output.json")
    # write_to_json5l(processed_data, json5l_path)

    # Step 5: 调用自动化音频合成脚本
    char_map_path = "char_to_voice_map.json"  # 假设文件固定，如有需要也可加入参数
    generate_and_run_audio_script(
        script_path=json5l_path,
        char_map_path=char_map_path,
        output_dir=args.output_path,
        result_filename="final_mix"
    )
