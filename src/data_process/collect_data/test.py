# Virus_GENBANK_accession = "UV21456"
# accession_list = []
# if ';' not in Virus_GENBANK_accession:
#     accession_list = [Virus_GENBANK_accession]
# else:
#     info_list = Virus_GENBANK_accession.split(';')
#     for info in info_list:
#         accession_list.append(info.split(':')[-1].strip())
# print(accession_list)

from fetch_data import FetchData

fetch_data = FetchData()
gb = fetch_data(': EU487045')
print(gb)