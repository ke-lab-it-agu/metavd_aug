from pathlib import Path
import sys
sys.path.append('./')
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init
import glob
import os
import pandas as pd
from openai import OpenAI
import openai
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

# OpenAI APIキーの設定
openai.api_key = "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"
os.environ["OPENAI_API_KEY"] = openai.api_key

def normalize_text(text): #テキストの正規化処理
    
    text = text.lower()  # 小文字化
    text = re.sub(r'[_\s]+', ' ', text)  # アンダースコアと空白の統一
    return text.strip()

def lemmatize_text(text): #テキストをレンマ化（単語の原形化
    
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)  # トークン化
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

def read_csv_files(directory): #指定されたディレクトリ内のすべてのCSVファイルを読み込み、動作ラベルを収集
   
    csv_files = glob.glob(f"{directory}/*.csv")
    action_labels = set()
    
    for file in csv_files:
        df = pd.read_csv(file)
        if 'name' in df.columns:  # 動作ラベルが保存された列名
            action_labels.update(df['name'].dropna().unique())
    
    return action_labels

def extract_action_labels_with_llm(description, action_labels): #LLMを利用してキャプションから動作ラベルを抽出

    client = OpenAI(
        api_key="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" # This is the default and can be omitted
    )
    
    prompt = f"""
                以下の動画キャプションを読み、動画の内容を正確に理解した上で、該当するすべての動作ラベルを選んでください。
                動作ラベルの選択の際、必ず、動作ラベルリストの中から該当する動作ラベルを選択してください。
                動作ラベルリスト内に存在しない語は生成してはいけません。

                【条件】
                1. 動作ラベルリストは動画内で行われている行動を表しています。
                2. 各ラベルの定義を考慮し、キャプションの具体的な内容に最も一致するラベルを選んでください。
                3. 動作ラベルが複数該当する場合は、カンマ区切りで列挙してください。
                4. 動作ラベルが全く該当しない場合は、「None」と記載してください。

                【動作ラベルの定義】
                - run: 人や動物が走っている行為
                - jump: 跳躍やジャンプしている行為
                - playing_rubiks_cube: ルービックキューブをする
                - Making a sandwich: サンドウィッチを作る
                ...

                【動画キャプション】
                {description}

                【動作ラベルリスト】
                {', '.join(action_labels)}

                【出力フォーマット】
                動作ラベル: <一致する動作ラベル（カンマ区切り）またはNone>

                """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # または使用可能なモデル名
            messages=[
                {"role": "system", "content": "You are a helpful assistant specialized in extracting actions."},
                {"role": "user", "content": prompt}
            ]
        )
        output = response.choices[0].message.content.strip()

        if "動作ラベル:" in output:
            labels = output.split("動作ラベル:")[1].strip()
            extracted_labels = [label.strip() for label in labels.split(",") if label.strip().lower() != "none"]
            # 動作ラベルリストに存在しないラベルを除外
            valid_labels = [label for label in extracted_labels if label in action_labels]
            return valid_labels
        else:
            return []
    except Exception as e:
        print(f"Error in LLM inference: {e}")
        return []

def inference():
    disable_torch_init()
    
    database_name = 'hmdb51'
    action_label = 'sword'
    video_name = '2006_Full_Contact_Medieval_Sword_Tournament_Final_sword_f_cm_np2_fr_bad_2.avi'
    """
    動画参照方法
    hmdb51  video_files = glob.glob(f'/disk1/{database_name}/{action_label}/{video_name} or *.avi')
    ucf101  video_files = glob.glob(f'/disk1/{database_name}/*{action_label}*.avi')
    stair_actions   video_files = glob.glob(f'/disk1/{database_name}/STAIR_Actions_v1.1/{action_label}/*.mp4')

    outputファイルの保存方法
    hmdb51  output_dir = Path(f"output/{database_name}/{action_label}")
    ucf101  output_dir = Path(f"output/{database_name}/{action_label}") 
    stair_actions   output_dir = Path(f"output/{database_name}/{action_label}")
    """

    video_files = glob.glob(f'/disk1/{database_name}/{action_label}/{video_name}.avi')
    instruct = f"""
                Generate a caption for the video that focuses solely on observable human actions. 
                Avoid describing emotions, intentions, or background context. Only state verifiable facts. 
                """
    
    model_path = 'DAMO-NLP-SG/VideoLLaMA2.1-7B-16F'
    model, processor, tokenizer = model_init(model_path)

    output_dir = Path(f"output/{database_name}/{action_label}")
    output_dir.mkdir(parents=True, exist_ok=True)

    max_output_length = 1000
    csv_directory = "/disk1/action_label_list"
    all_action_labels = read_csv_files(csv_directory)

    for modal_path in video_files:
        raw_output = mm_infer(processor['video'](modal_path), instruct, model=model, tokenizer=tokenizer, do_sample=False, modal='video')

        if len(raw_output) > max_output_length:
            raw_output = raw_output[:max_output_length] + "..."

        print(f'caption: {modal_path}:{raw_output}\n')

        matching_labels = extract_action_labels_with_llm(raw_output, all_action_labels)
        print(f'action labels: {matching_labels}\n')

        output_file = output_dir / f"{Path(modal_path).stem}.txt"
        with output_file.open("w") as f:
            f.write(f"caption:\n{raw_output}\n\n")
            f.write(f"action labels:\n{matching_labels}\n")

if __name__ == "__main__":
    inference()
