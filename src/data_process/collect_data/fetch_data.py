import requests
from io import StringIO
from Bio import SeqIO
# Bio.Entrez

class FetchData:
    """
    获取和处理数据并以 Dict 的形式返回
    """
    
    def fetch_genbank(self, accession: str, timeout: float = 15) -> str:
        """
        从 NCBI 下载 GenBank 格式的记录
        返回 GenBank 格式的文本内容
        timeout: 最大等待时间（秒），超过则抛出异常
        """
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            "db": "nuccore",
            "id": accession,
            "rettype": "gb",
            "retmode": "text"
        }
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()  # HTTP 错误抛出异常
            return r.text
        except requests.exceptions.Timeout:
            raise TimeoutError(f"请求超时：{accession} 在 {timeout} 秒内未返回")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"请求 {accession} 失败: {e}")
    
    def extract_cds_sequences(self, genbank_text: str) -> list[dict]:
        """
        从 GenBank 文本中提取每个 CDS 的核苷酸序列
        返回 list[dict]，列包括：
            - cds_id
            - location
            - sequence
        """
        handle = StringIO(genbank_text)
        records = list(SeqIO.parse(handle, "genbank"))
        rows = []
        for record in records:
            seq_full = record.seq
            for feature in record.features:
                if feature.type == "CDS":
                    loc = feature.location
                    try:
                        cds_seq = loc.extract(seq_full)
                    except Exception as e:
                        print(f"Warning: 无法提取位置 {loc} 的 CDS: {e}")
                        continue

                    qualifiers = feature.qualifiers
                    if "gene" in qualifiers:
                        cds_id = qualifiers["gene"][0]
                    elif "locus_tag" in qualifiers:
                        cds_id = qualifiers["locus_tag"][0]
                    elif "protein_id" in qualifiers:
                        cds_id = qualifiers["protein_id"][0]
                    else:
                        cds_id = str(loc)

                    rows.append({
                        "cds_id": cds_id,
                        "location": str(loc),
                        "sequence": str(cds_seq)
                    })

        return rows
    
    def __call__(self, accession: str, timeout: float = 10.0):
        genbank = self.fetch_genbank(accession, timeout=timeout)
        cds = self.extract_cds_sequences(genbank)
        return cds
