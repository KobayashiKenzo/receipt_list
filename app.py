import pandas as pd
import streamlit as st
import tempfile
from pathlib import Path
from main import MedicalReceiptProcessor

st.set_page_config(page_title="医療費明細書自動入力システム", layout="wide")
# perfects!
def main():
    st.title("医療費明細書自動入力システム")

    # ファイルアップロード
    uploaded_files = st.file_uploader("領収書PDFまたは画像をアップロード", type=["pdf", "png", "jpg"], accept_multiple_files=True)

    if uploaded_files:
        st.subheader("アップロードされたファイル一覧")
        file_data = pd.DataFrame({
            "ファイル名": [uploaded_file.name for uploaded_file in uploaded_files],
            "サイズ (KB)": [round(len(uploaded_file.getvalue()) / 1024, 2) for uploaded_file in uploaded_files]
        })
        st.dataframe(file_data)

        all_data = pd.DataFrame()  # 結果を統合するための空DataFrame

        # `phonebook.csv`のパスを指定（main.pyと同じディレクトリの場合）
        processor = MedicalReceiptProcessor('template/iryouhi_form_v3.xlsx', 'phonebook.csv')

        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                temp_file_path = tmp.name

            try:
                df = processor.extract_data(temp_file_path)
                st.write(f"【{uploaded_file.name}】の処理結果")
                columns_to_display = ['patient', 'institution', 'date', 'amount', 'categories']
                st.dataframe(df[columns_to_display])

                # 重複チェック
                all_data = pd.concat([all_data, df]).drop_duplicates(subset=['date', 'amount'], ignore_index=True)
            except Exception as e:
                st.error(f"{uploaded_file.name} の処理中にエラーが発生しました: {str(e)}")

        st.subheader("全てのデータを統合した結果")
        if not all_data.empty:
            st.dataframe(all_data)

        if st.button("Excelにまとめて反映"):
            output_file = Path("data/output/FY2024_Medical_Receipts.xlsx").resolve()
            if processor.update_excel(all_data, str(output_file)):
                st.success("Excelファイルが正常に更新されました！")
            else:
                st.error("Excel更新中にエラーが発生しました！")

if __name__ == "__main__":
    main()
