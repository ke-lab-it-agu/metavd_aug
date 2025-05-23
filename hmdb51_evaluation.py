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
from openpyxl import Workbook

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

# OpenAI APIキーの設定
openai.api_key = "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"
os.environ["OPENAI_API_KEY"] = openai.api_key

def normalize_text(text):
    """テキストの正規化処理"""
    text = text.lower()  # 小文字化
    text = re.sub(r'[_\s]+', ' ', text)  # アンダースコアと空白の統一
    return text.strip()

def lemmatize_text(text):
    """テキストをレンマ化（単語の原形化）"""
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)  # トークン化
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

def read_csv_files(directory):
    """指定されたディレクトリ内のすべてのCSVファイルを読み込み、動作ラベルを収集"""
    csv_files = glob.glob(f"{directory}/*.csv")
    action_labels = set()
    
    for file in csv_files:
        df = pd.read_csv(file)
        if 'name' in df.columns:  # 動作ラベルが保存された列名
            action_labels.update(df['name'].dropna().unique())
    
    return action_labels

def extract_action_labels_with_llm(description, action_labels):
    """LLMを利用してキャプションから動作ラベルを抽出"""
    client = OpenAI(
        api_key="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"
    )
    
    prompt = f"""
                以下の動画キャプションを読み、動画の内容を正確に理解した上で、該当するすべての動作ラベルを選んでください。
                動作ラベルの選択の際、必ず、動作ラベルリストの中から該当する動作ラベルを選択してください。
                動作ラベルリスト内に存在しない語は生成してはいけません。
                また、動画キャプションと元々付与されている動作ラベルが大きく異なる場合を除き、元々付与されていた動作ラベルは生成するようにしてください。

                【条件】
                1. 動作ラベルリストは動画内で行われている行動を表しています。
                2. 各ラベルの定義を考慮し、キャプションの具体的な内容に最も一致するラベルを選んでください。
                3. 動作ラベルが複数該当する場合は、カンマ区切りで列挙してください。
                4. 動作ラベルが全く該当しない場合は、「None」と記載してください。

                【動画キャプション】
                {description}

                【動作ラベルリスト】
                {', '.join(action_labels)}

                【出力フォーマット】
                動作ラベル: <一致する動作ラベル（カンマ区切り）またはNone>

                """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
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
    
def process_videos_hmdb51():
    database_name = 'hmdb51'
    base_dir = f'/disk1/{database_name}'


    csv_directory = f"/disk1/action_label_list"
    all_action_labels = read_csv_files(csv_directory)

    wb = Workbook()

    # モデルの初期化はループ外で行う
    model_path = 'DAMO-NLP-SG/VideoLLaMA2.1-7B-16F'
    model, processor, tokenizer = model_init(model_path)

    for action_label in os.listdir(base_dir):
        action_path = os.path.join(base_dir, action_label)
        if not os.path.isdir(action_path):
            continue

        video_files = glob.glob(f"{action_path}/*.avi")
        if not video_files:
            continue

        sheet = wb.create_sheet(title=action_label)
        sheet.append(["動画名", "動作ラベル"])

        output_dir = '/home/s-gotou/code/VideoLLaMA2/output'
        os.makedirs(output_dir, exist_ok=True)

         # Excelシートに書き込む
        for video_file in video_files:
            # 動画ごとにキャプションを生成
            instruct = f"""
                        Generate a caption for the video that focuses solely on observable human actions. 
                        Avoid describing emotions, intentions, or background context. Only state verifiable facts.
                        """
        

            # 動画からキャプションを生成
            raw_output = mm_infer(processor['video'](video_file), instruct, model=model, tokenizer=tokenizer, do_sample=False, modal='video')

            # キャプションの長さ制限
            max_output_length = 1000
            if len(raw_output) > max_output_length:
                raw_output = raw_output[:max_output_length] + "..."
            
            # 動作ラベルをLLMを使って抽出
            matching_labels = extract_action_labels_with_llm(raw_output, all_action_labels)
            print(f'action labels: {matching_labels}\n')
            sheet.append([os.path.basename(video_file), ', '.join(matching_labels) if matching_labels else 'None'])


    output_file = os.path.join(output_dir, database_name + '_evaluation.xlsx')
    wb.save(output_file)
    print(f"Excelファイルが保存されました: {output_file}")



if __name__ == "__main__":
    process_videos_hmdb51()
