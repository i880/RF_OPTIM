import argparse
import os
import pandas as pd
import zipfile
from glob import glob

class GSMWorstCells:
    def __init__(self, path):
        self.path = path

    def process(self):
        print("Processing GSM Worst Cells")
        os.chdir(self.path)
        for file in os.listdir(self.path):
            if zipfile.is_zipfile(file):
                with zipfile.ZipFile(file) as item:
                    item.extractall()

        all_files = glob('*.xlsx')
        sheets = pd.ExcelFile(all_files[0]).sheet_names
        dfs = {s: pd.concat(pd.read_excel(f, sheet_name=s) for f in all_files) for s in sheets}

        for filename in os.listdir(self.path):
            if filename.endswith('.xlsx'):
                os.unlink(os.path.join(self.path, filename))
        
        with pd.ExcelWriter('output_gsm.xlsx') as writer:
            for sheet, df in dfs.items():
                df.to_excel(writer, sheet_name=sheet, index=False)

class UMTSWorstCells:
    def __init__(self, path):
        self.path = path

    def process(self):
        print("Processing UMTS Worst Cells")
        os.chdir(self.path)
        for file in os.listdir(self.path):
            if zipfile.is_zipfile(file):
                with zipfile.ZipFile(file) as item:
                    item.extractall()

        all_files = glob('*.xlsx')
        sheets = pd.ExcelFile(all_files[0]).sheet_names
        dfs = {s: pd.concat(pd.read_excel(f, sheet_name=s) for f in all_files) for s in sheets}

        for filename in os.listdir(self.path):
            if filename.endswith('.xlsx'):
                os.unlink(os.path.join(self.path, filename))
        
        with pd.ExcelWriter('output_umts.xlsx') as writer:
            for sheet, df in dfs.items():
                df.to_excel(writer, sheet_name=sheet, index=False)

def main():
    parser = argparse.ArgumentParser(description="RF Optimization and Planning Tool")
    parser.add_argument('--gsm', help='Process GSM worst cells', action='store_true')
    parser.add_argument('--umts', help='Process UMTS worst cells', action='store_true')

    args = parser.parse_args()

    if args.gsm:
        gsm_processor = GSMWorstCells("/path/to/gsm")
        gsm_processor.process()

    if args.umts:
        umts_processor = UMTSWorstCells("/path/to/umts")
        umts_processor.process()

if __name__ == "__main__":
    main()
