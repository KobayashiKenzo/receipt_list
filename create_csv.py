import pandas as pd

# 金額関連キーワードのリスト
amount_keywords = [
    "請求額",
    "領収額",
]

# DataFrameを作成
df = pd.DataFrame({"receipts_words": amount_keywords})

# CSVファイルに保存
df.to_csv("amount_words.csv", index=False, encoding="utf-8-sig")
print("amount_words.csvを作成しました。")