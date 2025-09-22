from fetch_data import FetchData
import pandas as pd
from pathlib import Path
import csv
from tqdm import tqdm

def cds_record_generator(row, fetch_data,err_file):
    """
    对单行生成所有 CDS 记录，如果 accession 获取不到则跳过
    从数据库设计的角度来看，其实这里只需要增加Isolate ID就好了，用到什么信息直接做联表查询
    """
    # 对一个病毒携带多个链的情况，其实不难判断
    # 所有的accession都被卡在一个分号和一个冒号之间，直接分词即可
    Virus_GENBANK_accession = f"{row.Virus_GENBANK_accession}"
    accession_list = []
    if ';' not in Virus_GENBANK_accession:
        accession_list = [Virus_GENBANK_accession]
    else:
        info_list = Virus_GENBANK_accession.split(';')
        for info in info_list:
            accession_list.append(info.split(':')[-1].strip())

    for accession in accession_list:
        try:
            genbank = fetch_data.fetch_genbank(accession)
            if not genbank:
                # 返回空列表，也记录到错误文件
                print(f"[WARN] No gb found for accession: {accession}")
                err_file.write(f"No gb found: {accession}\n")
                continue
        except Exception as e:
            print(f"[ERROR] Failed to fetch accession {accession}: {e}")
            err_file.write(f"Fetch failed: {accession} | {e}\n")
            continue

        yield {
            "Isolate_ID": row.Isolate_ID,
            "accession": accession,
            "genbank": genbank
        }

def main():
    fetch_data = FetchData()
    input_file = "VMR_MSL40.v1.20250307.xlsx"
    err_file_path = "err_info.txt"
    data_dir = Path("./data")
    # 读取整个 Excel（小文件可行）
    df = pd.read_excel(input_file, sheet_name="VMR MSL40")
    df.columns = df.columns.str.strip().str.replace(r"\s+", "_", regex=True)

    if not data_dir.exists():
        data_dir.mkdir()
    # 打开 CSV 文件写入
    with open(err_file_path, "w", encoding="utf-8") as err_file:

        # 用 tqdm 添加进度条
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
            for record in cds_record_generator(row, fetch_data,err_file):
                record_file_path = data_dir / f'{record["Isolate_ID"]}.gb'
                with open(record_file_path,"a",encoding="utf-8") as f:
                    f.write(record["genbank"])

if __name__ == "__main__":
    main()
