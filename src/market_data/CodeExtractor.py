import pathlib
import pandas as pd

def extract_topix_codes():
    parent_dir = pathlib.Path(__file__).parent.parent.parent
    code_data_dir = parent_dir / "data/raw/code"
    file_path = code_data_dir / "topixweight_j.csv"

    df = pd.read_csv(file_path, encoding='shift_jis')
    print(df.head())

    # Core30, Large70のコードを取得
    filtered_df = df[df["ニューインデックス区分"].isin(["TOPIX Core30", "TOPIX Large70"])]
    codes = [str(code) + ".T" for code in filtered_df["コード"].tolist()]
    return codes

if __name__ == "__main__":
    codes = extract_topix_codes()
    print(f"取得したコード数: {len(codes)}")
    print(codes)