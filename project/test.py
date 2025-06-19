from llama_cpp import Llama
import config
import re

# モデルをロード
model_path = config.get_model_path()
llm = Llama(
        model_path=model_path,
        chat_format="llama-3",
        n_ctx=16384,
        n_threads=8,
        n_gpu_layers=-1,
        verbose=False,
    )

# ストリーミングを有効にしてチャットリクエスト
stream = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "you are a helpful assistant."},
        {"role": "user", "content": "Members of which of the following groups are most likely to be surrounded by a protein coat?\nA. Viruses\nB. Bacteria\nC. Fungi\nD. Plants"},
        # {"role": "user", "content": "Output in JSON as:{reasoning: <Why you chose that answer>, answer: <A/B/C/D>}"},
    ],
    stream=True  # ここでストリーミングを有効にする
)


accumulated_text = ""
count = 0

for chunk in stream:
    delta = chunk['choices'][0]['delta']
    
    if 'content' in delta:
        # 新しいトークンを蓄積
        accumulated_text += delta['content']
        
        # 文の区切り文字をチェック（英語・日本語対応）
        while True:
            # 文の区切りを検索
            match = re.search(r'[.!?。！？,]+', accumulated_text)
            if match:
                # 区切り文字の位置を取得
                end_pos = match.end()
            
                sentence = accumulated_text[:end_pos].strip()
                if sentence:
                    count += 1
                    print(sentence)
                    print()  # 改行
                
                # 表示した文を除去
                accumulated_text = accumulated_text[end_pos:].strip()
                # if count >= 3:
                #     stream.close()  # 3文表示したらストリームを閉じる
                #     # 3文表示したら終了
                #     break
            else:
                # 完全な文がない場合はループを抜ける
                break

# 最後に残った文があれば表示
if accumulated_text.strip():
    print(accumulated_text.strip())