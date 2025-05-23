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

def normalize_text(text):
    """テキストの正規化処理"""
    text = text.lower()
    text = re.sub(r'[_\s]+', ' ', text)
    return text.strip()

def lemmatize_text(text):
    """テキストをレンマ化（単語の原形化）"""
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

def read_csv_files(directory):
    """指定されたディレクトリ内のすべてのCSVファイルを読み込み、動作ラベルを収集"""
    csv_files = glob.glob(f"{directory}/*.csv")
    action_labels = set()
    
    for file in csv_files:
        df = pd.read_csv(file)
        if 'name' in df.columns:
            action_labels.update(df['name'].dropna().unique())
    
    return action_labels

def extract_action_labels_with_llm(description, action_labels):
    """LLMを利用してキャプションから動作ラベルを抽出"""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    prompt = f"""
                以下の動画キャプションを読み、適切な動作ラベルをリストから選んでください。
                動作ラベルリスト内の語のみを選択してください。

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
    

def add_equal_action_labels(action_labels, csv_path):
    """ 
    extract_action_labels_with_llm で抽出された動作ラベルに対して、
    MetaVDの情報からequal関係にある動作ラベルを追加する
    """
    df = pd.read_csv(csv_path)

    # 動作ラベルのセット（重複を避けるため）
    updated_labels = set(action_labels)

    for _, row in df.iterrows():
        if row['relation'] == 'equal':  # relation が equal の場合のみ処理
            from_label = row['from_action_name']
            to_label = row['to_action_name']

            # 抽出されたラベルと一致するものがあるか確認
            if from_label in action_labels and to_label not in updated_labels:
                updated_labels.add(to_label)
            elif to_label in action_labels and from_label not in updated_labels:
                updated_labels.add(from_label)

    return list(updated_labels)


def process_videos_hmdb51():
    database_name = 'hmdb51'
    base_dir = f'/disk1/{database_name}'
    
    csv_directory = f"/disk1/action_label_list"
    all_action_labels = read_csv_files(csv_directory)
    
    output_dir = '/home/s-gotou/code/VideoLLaMA2/output/evaluation/hmdb51'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{database_name}_evaluation.csv")
    
    model_path = 'DAMO-NLP-SG/VideoLLaMA2.1-7B-16F'
    model, processor, tokenizer = model_init(model_path)
    
    video_files = glob.glob(f"{base_dir}/*/*.avi")
    
    with open(output_file, 'w') as f:
        f.write("動画名,動作ラベル\n")
        
        for video_file in video_files:
            # 1. 動画キャプションを生成
            instruct = f"""
                        Generate a caption for the video that focuses solely on observable human actions. 
                        Avoid describing emotions, intentions, or background context. Only state verifiable facts.
                        """
            raw_output = mm_infer(processor['video'](video_file), instruct, model=model, tokenizer=tokenizer, do_sample=False, modal='video')
            
            # 2. 動画キャプションと動作ラベルリストから適したラベルを抽出
            matching_labels = extract_action_labels_with_llm(raw_output, all_action_labels)
            
            # 3. MetaVDの情報を元に追加の動作ラベルを取得
            extended_labels = add_equal_action_labels(matching_labels, '/disk1/metavd_relation_list/metavd_v1_equal_ex.csv')
            print(f'action labels: {extended_labels}\n')
            
            # 4. 結果をCSVに書き込む
            f.write(f"{os.path.basename(video_file)},{', '.join(extended_labels) if extended_labels else 'None'}\n")
    
    print(f"CSVファイルが保存されました: {output_file}")

if __name__ == "__main__":
    process_videos_hmdb51()  
