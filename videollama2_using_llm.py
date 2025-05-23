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
openai.api_key = "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"
os.environ["OPENAI_API_KEY"] = openai.api_key
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')



def normalize_text(text): #テキストの正規化処理。小文字化、単語間の空白とアンダースコアの統一
    
    text = text.lower()  # 小文字化
    text = re.sub(r'[_\s]+', ' ', text)  # アンダースコアと空白の統一
    return text.strip()

def lemmatize_text(text): #テキストをレンマ化（単語の原形化）
    
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)  # トークン化
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

def extract_matching_labels(raw_output, all_action_labels): #raw_output と all_action_labels の間で、正規化＋レンマ化による完全一致をチェック、最終的に all_action_labels の表記を保持
    
    normalized_output = normalize_text(raw_output)
    lemmatized_output = lemmatize_text(normalized_output)
    matching_labels = []

    for label in all_action_labels:
        normalized_label = normalize_text(label)
        lemmatized_label = lemmatize_text(normalized_label)
        
        # 正規化された一致チェック
        if re.search(fr'\b{re.escape(normalized_label)}\b', normalized_output):
            matching_labels.append(label)

        # レンマ化された一致チェック
        elif re.search(fr'\b{re.escape(lemmatized_label)}\b', lemmatized_output):
            matching_labels.append(label)
    return matching_labels


def read_csv_files(directory): #指定されたディレクトリ内のすべてのCSVファイルを読み込み、動作ラベルを収集

    csv_files = glob.glob(f"{directory}/*.csv")
    action_labels = set()
    
    for file in csv_files:
        # 各CSVファイルを読み込み
        df = pd.read_csv(file)
        # 必要に応じて列名を調整
        if 'name' in df.columns:  # 動作ラベルが保存された列名
            action_labels.update(df['name'].dropna().unique())
    
    return action_labels

# 動作ラベルを抽出する関数（GPT-4o-miniを使用）
def extract_action_labels(description,action_labels):
    # 外部LLMを利用して動作ラベルを抽出する処理
    # ここでは仮に description を受け取って動作ラベルを返す処理を実装します
    # OpenAIのAPIを利用する例
    client = OpenAI(
        api_key="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" # This is the default and can be omitted
    )
    

   # プロンプト生成
    prompt = f"""
            以下の文章を読み、指定された動作ラベルリストの中から一致するラベルを抽出してください。
            一致する動作ラベルは、以下の条件に基づいて抽出します：

            1. 一致する場合は、動作ラベルをそのまま表記してください。
            2. 一致する動作ラベルが複数ある場合は、カンマ区切りで列挙してください。
            3. 一致するものがない場合は、「None」と記載してください。

            【文章】
            {description}

            【動作ラベルリスト】
            {', '.join(action_labels)}

            【出力フォーマット】
            動作ラベル: <一致するラベル>
    """

    
     # 新しいChatCompletion APIを使用
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant specialized in extracting actions."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def inference():
    disable_torch_init()
    video_name = 'APOCALYPTO_eat_u_nm_np1_fr_goo_6'
    database_name = 'hmdb51'
    action_label = 'eat'

    # 動画ファイルの指定
    video_files = glob.glob('/disk1/'+database_name+'/'+action_label+'/'+video_name+'.avi')
    
    # 入力 
    instruct = "Watch this video and state the fact of the video with attention to the action"
    
    # モデルの指定
    model_path = 'DAMO-NLP-SG/VideoLLaMA2.1-7B-16F'
    model, processor, tokenizer = model_init(model_path)

    # 保存先ディレクトリの指定
    output_dir = Path("output/"+database_name+"/"+action_label)

    # ディレクトリが存在しない場合は作成
    output_dir.mkdir(parents=True, exist_ok=True)

    # 出力文字数の制限
    max_output_length = 1000  # 最大文字数を指定

    csv_directory = "/disk1/action_label_list"  # 実際のCSVディレクトリ
    all_action_labels = read_csv_files(csv_directory)  # read_csv_files() 関数でラベル取得


    # ループ処理
    for modal_path in video_files:
        # 動画キャプションを生成
        raw_output = mm_infer(processor['video'](modal_path), instruct, model=model, tokenizer=tokenizer, do_sample=False, modal='video')

        # 文字数制限を適用
        if len(raw_output) > max_output_length:
            raw_output = raw_output[:max_output_length] + "..."  # 制限後に省略記号を追加

        print(f'caption: {modal_path}:\n{raw_output}\n')

        # 動作ラベルの抽出
        matching_labels = extract_matching_labels(raw_output, all_action_labels)
        print(f'action labels: {matching_labels}\n')

        # キャプションと動作ラベルをファイルに保存
        output_file = output_dir / f"{Path(modal_path).stem}.txt"
        with output_file.open("w") as f:
            f.write(f"caption:\n{raw_output}\n\n")
            f.write(f"action labels:\n{matching_labels}\n")


if __name__ == "__main__":
    inference()