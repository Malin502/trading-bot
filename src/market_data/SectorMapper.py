import pathlib
import pandas as pd
import json


def mapping_sector():
    parent_dir = pathlib.Path(__file__).parent.parent.parent
    code_data_dir = parent_dir / "data/raw/code"
    file_path = code_data_dir / "topixweight_j.csv"

    df = pd.read_csv(file_path, encoding='shift_jis')
    print(df.head())

    # Core30, Large70のコードを取得
    filtered_df = df[df["ニューインデックス区分"].isin(["TOPIX Core30", "TOPIX Large70"])]
    codes = [str(code) + ".T" for code in filtered_df["コード"].tolist()]
    sector = [str(sec) for sec in filtered_df["業種"].tolist()]
    
    sector_map = dict(zip(codes, sector))
    sector_map["1306.T"] = "Index"
    with open(parent_dir / "config/sector_mapping.json", "w", encoding="utf-8") as f:
        json.dump(sector_map, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(sector_map)} entries to {parent_dir / 'config/sector_mapping.json'}")
    
    return sector_map
    
if __name__ == "__main__":
    sector_map = mapping_sector()
    print(f"取得したセクターマップの件数: {len(sector_map)}")
    print(sector_map)
