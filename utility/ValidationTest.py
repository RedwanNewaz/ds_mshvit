import pandas as pd 
import os 

if __name__ == '__main__':
    filename = '/home/aredwann/MSHViT/annotations_sewerml/ori/SewerML_Test.csv'
    src_dir = '/usace_share/SewerML/valid01'
    csv_data = pd.read_csv(filename)
    countExist = 0 
    countNoExist = 0 
    for item in csv_data['Filename']:
        src_path = os.path.join(src_dir, item)
        if not os.path.exists(src_path):
            countNoExist += 1
            print(f"[!] {countNoExist}/{countExist} path does not exist = {src_path}", flush=True, end='\r')
        else:
            countExist += 1