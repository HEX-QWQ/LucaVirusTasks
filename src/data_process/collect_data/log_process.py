import pandas as pd

input_file = "VMR_MSL40.v1.20250307.xlsx"
no_cds_path = "no_cds.csv"
no_accession_path = "no_accession.csv"
df = pd.read_excel(input_file, sheet_name="VMR MSL40")
df.columns = df.columns.str.strip().str.replace(r"\s+", "_", regex=True)


# 记录没有cds号的情况
no_cds_record_list = []
with open("nohup.out","r",encoding="utf-8") as f:
    for line in f:
        if '[WARN] No CDS found for accession:' in line:
            accession = line.split(':')[-1].strip()
            if accession == 'nan':
                continue
            Isolate_ID = df[df["Virus_GENBANK_accession"].str.contains(accession, na=False,regex=False)]["Isolate_ID"].iloc[0]
            # print(type(Isolate_ID))
            no_cds_record_list.append({
                "Isolate_ID": Isolate_ID,
                "accession": accession
            })
            

no_cds_df = pd.DataFrame(no_cds_record_list)
no_cds_df.to_csv(no_cds_path,index=False)

# 记录accession号为空的情况
print("process 2")
no_accession_list = []

for _,row in df.iterrows():
    accession = f'{row.Virus_GENBANK_accession}'
    if accession == 'nan':
        no_accession_list.append({
            "Isolate_ID": row.Isolate_ID
        })

no_accession_df = pd.DataFrame(no_accession_list)
no_accession_df.to_csv(no_accession_path,index=False)


