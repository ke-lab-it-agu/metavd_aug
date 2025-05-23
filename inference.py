from pathlib import Path
import sys
sys.path.append('./')
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init
import glob


def inference():
    disable_torch_init()
    video_name = 'APOCALYPTO_eat_u_nm_np1_fr_goo_6'
    database_name = 'hmdb51'
    action_label = 'eat'

    # 動画ファイルの指定
    video_files = glob.glob('/disk1/'+database_name+'/'+action_label+'/*.avi')
    
    # 入力 
    instruct = 'Please list all actions in this video. Please output the following as an example, "eating cake and watching tv."'
    
    # モデルの指定
    model_path = 'DAMO-NLP-SG/VideoLLaMA2.1-7B-16F'
    model, processor, tokenizer = model_init(model_path)

    # 保存先ディレクトリの指定
    output_dir = Path("output/"+database_name+"/"+action_label)

    # ディレクトリが存在しない場合は作成
    output_dir.mkdir(parents=True, exist_ok=True)

    # 出力文字数の制限
    max_output_length = 200  # 最大文字数を指定

    
    # ループ処理
    for modal_path in video_files:
        output = mm_infer(processor['video'](modal_path), instruct, model=model, tokenizer=tokenizer, do_sample=False, modal='video')

         # 文字数制限を適用
        if len(output) > max_output_length:
            output = output[:max_output_length] + "..."  # 制限後に省略記号を追加

        print(f'Output for {modal_path}:\n{output}\n')
        
        # ファイルの書き込み
        output_file = output_dir / f"{Path(modal_path).stem}.txt"
        with output_file.open("w") as f:
            f.write(output)


if __name__ == "__main__":
    inference()