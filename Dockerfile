FROM python:3.12.3-slim

# 環境変数の設定（最初にやることで `RUN` でも使える）
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata
ENV PYTHONUNBUFFERED=1
ENV LANG=ja_JP.UTF-8
ENV LC_ALL=ja_JP.UTF-8

# 必要なパッケージのインストール（1回でまとめる）
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    wget \
    fonts-ipafont \
    locales \
    && locale-gen ja_JP.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

RUN sed -i 's/^# *\(ja_JP.UTF-8 UTF-8\)/\1/' /etc/locale.gen && \
    locale-gen
        

# Tesseract の日本語言語データをダウンロード（1回にまとめる）
RUN wget -P /usr/share/tesseract-ocr/4.00/tessdata/ \
    https://github.com/tesseract-ocr/tessdata_best/raw/main/jpn.traineddata \
    https://github.com/tesseract-ocr/tessdata_best/raw/main/jpn_vert.traineddata

# 作業ディレクトリを設定
WORKDIR /app

# 言語データ確認スクリプト（必要なら残す）
RUN echo '#!/bin/bash\n\
echo "Checking Tesseract language data..."\n\
echo "TESSDATA_PREFIX=$TESSDATA_PREFIX"\n\
ls -l $TESSDATA_PREFIX\n\
if [ -f ${TESSDATA_PREFIX}/jpn.traineddata ]; then\n\
    echo "Japanese language data found"\n\
else\n\
    echo "Japanese language data not found"\n\
    echo "Searching for jpn.traineddata..."\n\
    find / -name "jpn.traineddata" 2>/dev/null\n\
fi' > /check-lang.sh && chmod +x /check-lang.sh

# Python依存パッケージのインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションファイルとテンプレートファイルの配置
COPY app.py main.py ./
COPY template/iryouhi_form_v3.xlsx ./iryouhi_form_v3.xlsx
COPY phonebook.csv patients.csv /app/

# デバッグ情報を表示してからアプリを起動
CMD ["/bin/bash", "-c", "/check-lang.sh && streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]
