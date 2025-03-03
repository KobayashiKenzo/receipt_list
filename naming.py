import os
import sys
from pathlib import Path
from main import MedicalReceiptProcessor

def rename_pdf_files(directory_path: str, phonebook_path: str):
    """
    指定ディレクトリ内のPDFファイルを処理し、内容に基づいてリネームする
    """
    # MedicalReceiptProcessorのインスタンス化
    processor = MedicalReceiptProcessor('template/iryouhi_form_v3.xlsx', phonebook_path)
    
    # ディレクトリ内のPDFファイルを取得
    pdf_files = list(Path(directory_path).glob('*.pdf'))
    print(f"{len(pdf_files)}個のPDFファイルを処理します...")
    
    for pdf_path in pdf_files:
        try:
            # PDFからデータを抽出
            result_df = processor.extract_data(str(pdf_path))
            
            if not result_df.empty:
                # 必要な情報を取得
                date = result_df.iloc[0].get('date', '不明日付')
                institution = result_df.iloc[0].get('institution', '不明施設')
                amount = int(result_df.iloc[0].get('amount', 0))

                
                # ファイル名に使えない文字を置換
                institution = institution.replace('/', '').replace('\\', '').replace(':', '').replace('*', '').replace('?', '').replace('"', '').replace('<', '').replace('>', '').replace('|', '')
                
                # 新しいファイル名を生成
                print(f"抽出データ: date={date}, institution={institution}, amount={amount}, type={type(amount)}")
                new_filename = f"{institution}_{date}_{amount}.pdf"
                new_path = pdf_path.parent / new_filename
                
                # ファイル名が既に存在する場合の処理
                counter = 1
                while new_path.exists():
                    new_filename = f"{institution}_{date}_{amount}_{counter}.pdf"
                    new_path = pdf_path.parent / new_filename
                    counter += 1
                
                # ファイルをリネーム
                os.rename(pdf_path, new_path)
                print(f"リネーム成功: {pdf_path.name} → {new_filename}\n\n\n")
            else:
                print(f"データ抽出失敗: {pdf_path.name}\n\n\n")
        
        except Exception as e:
            print(f"処理エラー ({pdf_path.name}): {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python naming.py <PDFディレクトリパス> [phonebook.csvパス]")
        sys.exit(1)
    
    directory_path = sys.argv[1]
    phonebook_path = sys.argv[2] if len(sys.argv) > 2 else "phonebook.csv"
    
    rename_pdf_files(directory_path, phonebook_path)
