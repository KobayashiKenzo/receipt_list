import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import pandas as pd
import os
import glob
import fitz  # PyMuPDF (PDF to Image)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from rapidfuzz.distance import Levenshtein

print("Current Working Directory:", os.getcwd())


# ================================
# 1. データの読み込み
# ================================
# 正解データ（train_data.csv）
ground_truth = pd.read_csv("/app/data/train_data.csv")

# PDFファイルの一覧を取得
pdf_files = glob.glob("/app/data/*.pdf")
print("PDFファイルの一覧:", pdf_files)

# ================================
# 2. PDF → 画像変換（PyMuPDF を使用）
# ================================
def pdf_to_images(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images

# ================================
# 3. OCR用関数
# ================================
def process_image(image, contrast, sharpen, threshold, psm, oem):
    """ 画像処理とOCR """
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)
    if sharpen:
        image = image.filter(ImageFilter.SHARPEN)
    image = image.convert("L")  # グレースケール化
    image = image.point(lambda x: 0 if x < threshold else 255, '1')  # 二値化

    # OCR実行
    config = f"--psm {psm} --oem {oem}"
    text = pytesseract.image_to_string(image, lang="jpn", config=config)
    return text.strip()

# ================================
# 4. パラメータグリッド（試行するOCR設定）
# ================================
param_grid = {
    "contrast": [1.5, 2.0, 3.0],  # コントラスト強調
    "sharpen": [True, False],  # シャープネス
    "threshold": [150, 180, 200, 220],  # 二値化しきい値
    "psm": [3, 6, 11],  # Page Segmentation Mode
    "oem": [1, 3]  # OCR Engine Mode
}

# ================================
# 5. OCRパラメータ最適化
# ================================
X = []
y = []

for pdf_path in pdf_files:
    # PDF名と一致する正解データを取得
    pdf_name = os.path.basename(pdf_path)
    correct_row = ground_truth[ground_truth["filename"] == pdf_name]
    print("処理中のPDF:", pdf_name)
    print("対応する正解データ:", correct_row)

    if correct_row.empty:
        continue  # 該当するデータがない場合はスキップ

    correct_text = " ".join(map(str, correct_row.iloc[0, 1:].values))  # すべてのカラムを結合（患者名、金額、日付など）

    # PDF を画像に変換
    images = pdf_to_images(pdf_path)

    for image in images:
        for contrast in param_grid["contrast"]:
            for sharpen in param_grid["sharpen"]:
                for threshold in param_grid["threshold"]:
                    for psm in param_grid["psm"]:
                        for oem in param_grid["oem"]:
                            # OCR 実行
                            ocr_text = process_image(image, contrast, sharpen, threshold, psm, oem)

                            # Levenshtein距離（OCR誤差）
                            error_score = Levenshtein.normalized_distance(correct_text, ocr_text)

                            # データ保存
                            X.append([contrast, sharpen, threshold, psm, oem])
                            y.append(error_score)

# ================================
# 6. 機械学習による最適化
# ================================
X = pd.DataFrame(X, columns=["contrast", "sharpen", "threshold", "psm", "oem"])
y = np.array(y)

# データの中身を確認
print(X.info())  # 各カラムのデータ型や欠損値をチェック
print(X.isnull().sum())  # 欠損値の数を確認

print("Xのデータ数:", len(X))
print("yのデータ数:", len(y))

# 訓練データとテストデータに分割（データが空でないことを確認）
if len(X) > 0 and len(y) > 0:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("データ分割成功！")
else:
    print("データが空なので、train_test_split() を実行できません。")
    exit()  # データがない場合は処理を終了

# ランダムフォレスト回帰
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# テストデータで評価
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

# 最適なパラメータを取得
best_params = X.iloc[np.argmin(y)]
print("Best OCR Parameters:", best_params.to_dict())

# ================================
# 7. 結果をExcelに保存
# ================================

# 出力ディレクトリの作成
output_dir = "/app/data/output"
os.makedirs(output_dir, exist_ok=True)  # フォルダがなければ作成

# 出力パスを修正
output_path = os.path.join(output_dir, "ocr_optimization_results.xlsx")

# 最適パラメータをExcelに保存
best_params_df = pd.DataFrame([best_params])
best_params_df.to_excel(output_path, index=False)

print(f"✅ 最適なパラメータを {output_path} に保存しました！")

# ================================
# 8. テスト用PDF（20241130_玉川_760.pdf）のOCRを実行
# ================================

test_pdf = "/app/data/20241130_玉川_760.pdf"

print(f"🎯 テスト用PDF {test_pdf} に最適なOCR設定を適用")

# PDF を画像に変換
test_images = pdf_to_images(test_pdf)

# 最適パラメータを取得
best_contrast = best_params["contrast"]
best_sharpen = best_params["sharpen"]
best_threshold = best_params["threshold"]
best_psm = best_params["psm"]
best_oem = best_params["oem"]

# OCR 実行
test_results = []
for image in test_images:
    ocr_text = process_image(image, best_contrast, best_sharpen, best_threshold, best_psm, best_oem)
    test_results.append(ocr_text)

# OCR結果を出力
print("🔍 テスト用PDFのOCR結果:")
for i, text in enumerate(test_results):
    print(f"--- Page {i+1} ---")
    print(text)
    print("-----------------")

# OCR結果をファイルに保存
test_output_path = os.path.join(output_dir, "test_ocr_results.txt")
with open(test_output_path, "w", encoding="utf-8") as f:
    for i, text in enumerate(test_results):
        f.write(f"--- Page {i+1} ---\n")
        f.write(text + "\n")
        f.write("-----------------\n")

print(f"✅ テスト用PDFのOCR結果を {test_output_path} に保存しました！")
