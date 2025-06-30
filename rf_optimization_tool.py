#!/usr/bin/env python3
"""
RF Optimization and Planning Tool
Based on "Python For RF Optimization & Planning Engineers" 

This tool provides various RF optimization and planning functionalities
for GSM, UMTS, and LTE networks.
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import zipfile
from glob import glob
import warnings
warnings.filterwarnings('ignore')

class RFOptimizationTool:
    """Main class for RF Optimization Tool"""
    
    def __init__(self, working_directory="."):
        self.working_directory = working_directory
        self.original_directory = os.getcwd()
        
    def change_working_directory(self, path):
        """Change working directory safely"""
        if os.path.exists(path):
            os.chdir(path)
            self.working_directory = path
            print(f"Working directory changed to: {path}")
        else:
            print(f"Directory {path} does not exist. Creating it...")
            os.makedirs(path, exist_ok=True)
            os.chdir(path)
            self.working_directory = path
            
    def restore_directory(self):
        """Restore original working directory"""
        os.chdir(self.original_directory)

class GSMWorstCellProcessor(RFOptimizationTool):
    """Process GSM Worst Cells from PRS Reports"""
    
    def __init__(self, working_directory="./gsm_worst_cells"):
        super().__init__(working_directory)
        
    def extract_zip_files(self):
        """Extract all zip files in the working directory"""
        print("Extracting ZIP files...")
        for file in os.listdir(self.working_directory):
            if zipfile.is_zipfile(file):
                with zipfile.ZipFile(file) as item:
                    item.extractall()
                print(f"Extracted: {file}")
                
    def merge_excel_files(self):
        """Merge all Excel files and tabs"""
        print("Merging Excel files...")
        all_files = glob('*.xlsx')
        if not all_files:
            print("No Excel files found!")
            return
            
        sheets = pd.ExcelFile(all_files[0]).sheet_names
        dfs = {}
        
        for sheet in sheets:
            sheet_dfs = []
            for file in all_files:
                try:
                    df = pd.read_excel(file, sheet_name=sheet,
                                     converters={'Integrity': lambda value: '{:,.0f}%'.format(value * 100) if isinstance(value, (int, float)) else value})
                    sheet_dfs.append(df)
                except Exception as e:
                    print(f"Error reading {file}, sheet {sheet}: {e}")
                    
            if sheet_dfs:
                dfs[sheet] = pd.concat(sheet_dfs, ignore_index=True)
                
        return dfs
        
    def cleanup_temp_files(self):
        """Remove temporary Excel files"""
        for filename in os.listdir(self.working_directory):
            if filename.endswith('.xlsx') and not filename.startswith('output'):
                os.unlink(os.path.join(self.working_directory, filename))
                
    def export_results(self, dfs):
        """Export final dataset"""
        print("Exporting results...")
        with pd.ExcelWriter('2G_WC_Output.xlsx', engine='openpyxl') as writer:
            for sheet_name, df in dfs.items():
                df.to_excel(writer, sheet_name=sheet_name[:31], index=False)  # Excel sheet name limit
        print("GSM Worst Cells processing completed: 2G_WC_Output.xlsx")
        
    def process(self):
        """Main processing function"""
        self.change_working_directory(self.working_directory)
        try:
            self.extract_zip_files()
            dfs = self.merge_excel_files()
            if dfs:
                self.cleanup_temp_files()
                self.export_results(dfs)
            else:
                print("No data to process!")
        finally:
            self.restore_directory()

class UMTSWorstCellProcessor(RFOptimizationTool):
    """Process UMTS Worst Cells from PRS Reports"""
    
    def __init__(self, working_directory="./umts_worst_cells"):
        super().__init__(working_directory)
        
    def extract_zip_files(self):
        """Extract all zip files in the working directory"""
        print("Extracting ZIP files...")
        for file in os.listdir(self.working_directory):
            if zipfile.is_zipfile(file):
                with zipfile.ZipFile(file) as item:
                    item.extractall()
                print(f"Extracted: {file}")
                
    def merge_excel_files(self):
        """Merge all Excel files and tabs"""
        print("Merging Excel files...")
        all_files = glob('*.xlsx')
        if not all_files:
            print("No Excel files found!")
            return
            
        sheets = pd.ExcelFile(all_files[0]).sheet_names
        dfs = {}
        
        for sheet in sheets:
            sheet_dfs = []
            for file in all_files:
                try:
                    df = pd.read_excel(file, sheet_name=sheet,
                                     converters={'Integrity': lambda value: '{:,.0f}%'.format(value * 100) if isinstance(value, (int, float)) else value})
                    sheet_dfs.append(df)
                except Exception as e:
                    print(f"Error reading {file}, sheet {sheet}: {e}")
                    
            if sheet_dfs:
                dfs[sheet] = pd.concat(sheet_dfs, ignore_index=True)
                
        # Add blank columns for UMTS specific processing
        for sheet_name, df in dfs.items():
            if 'CSSR' in sheet_name or 'RRC SSR' in sheet_name:
                df["Comments"] = ''
                df["Bottleneck"] = ''
                df["Status"] = ''
                
        return dfs
        
    def cleanup_temp_files(self):
        """Remove temporary Excel files"""
        for filename in os.listdir(self.working_directory):
            if filename.endswith('.xlsx') and not filename.startswith('output'):
                os.unlink(os.path.join(self.working_directory, filename))
                
    def export_results(self, dfs):
        """Export final dataset"""
        print("Exporting results...")
        with pd.ExcelWriter('3G_WCELL_Output.xlsx', engine='openpyxl') as writer:
            for sheet_name, df in dfs.items():
                df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        print("UMTS Worst Cells processing completed: 3G_WCELL_Output.xlsx")
        
    def process(self):
        """Main processing function"""
        self.change_working_directory(self.working_directory)
        try:
            self.extract_zip_files()
            dfs = self.merge_excel_files()
            if dfs:
                self.cleanup_temp_files()
                self.export_results(dfs)
            else:
                print("No data to process!")
        finally:
            self.restore_directory()

class HexDecimalConverter(RFOptimizationTool):
    """Convert LAC TAC values from Hexadecimal to Decimal"""
    
    def __init__(self, working_directory="./hex_to_dec", input_file="TACLAC.txt"):
        super().__init__(working_directory)
        self.input_file = input_file
        
    def process(self):
        """Convert hex values to decimal"""
        self.change_working_directory(self.working_directory)
        try:
            print("Converting Hexadecimal to Decimal...")
            
            # Check if input file exists
            if not os.path.exists(self.input_file):
                print(f"Input file {self.input_file} not found!")
                return
                
            df = pd.read_csv(self.input_file)
            
            # Convert TAC and LAC from hex to decimal
            if 'TAC LAC' in df.columns:
                df['TAC'] = df['TAC LAC'].str.split(' ').str[0].apply(lambda x: int(x, 16) if pd.notna(x) else None)
                df['LAC'] = df['TAC LAC'].str.split(' ').str[2].apply(lambda x: int(x, 16) if pd.notna(x) and len(df['TAC LAC'].str.split(' ').iloc[0]) > 2 else None)
            
            # Export results
            df.to_csv('Final_Values.csv', index=False)
            print("Conversion completed: Final_Values.csv")
            
        except Exception as e:
            print(f"Error during conversion: {e}")
        finally:
            self.restore_directory()

class ClusterBusyHourCalculator(RFOptimizationTool):
    """Calculate Cluster Busy Hour from hourly traffic data"""
    
    def __init__(self, working_directory="./cluster_bh"):
        super().__init__(working_directory)
        
    def process_separate_datetime(self):
        """Process files where Date and Time are in separate columns"""
        print("Processing files with separate Date and Time columns...")
        
        files = sorted(glob('*.csv'))
        if not files:
            print("No CSV files found!")
            return
            
        df_list = []
        for file in files:
            try:
                df = pd.read_csv(file, header=3, skipfooter=1, engine='python', 
                               na_values=['NIL', '/0'], parse_dates=["Date"])
                df_list.append(df)
            except Exception as e:
                print(f"Error reading {file}: {e}")
                
        if df_list:
            df_combined = pd.concat(df_list, ignore_index=True).sort_values('Date')
            
            # Calculate busy hour (max traffic per cluster per day)
            if 'GlobelTraffic' in df_combined.columns and 'GCell Group' in df_combined.columns:
                df_bh = df_combined.loc[df_combined.groupby(['Date', 'GCell Group'])['GlobelTraffic'].idxmax()]
                df_bh.to_csv('cluster_bh_output.csv', index=False)
                print("Cluster Busy Hour calculation completed: cluster_bh_output.csv")
            else:
                print("Required columns (GlobelTraffic, GCell Group) not found!")
                
    def process_combined_datetime(self):
        """Process files where Date and Time are in the same column"""
        print("Processing files with combined Date and Time column...")
        
        files = sorted(glob('*.csv'))
        if not files:
            print("No CSV files found!")
            return
            
        df_list = []
        for file in files:
            try:
                df = pd.read_csv(file, header=3, skipfooter=1, engine='python', 
                               na_values=['NIL', '/0'], parse_dates=["Time"])
                df_list.append(df)
            except Exception as e:
                print(f"Error reading {file}: {e}")
                
        if df_list:
            df_combined = pd.concat(df_list, ignore_index=True).sort_values('Time')
            
            # Split Date and Time
            df_combined['Date'] = pd.to_datetime(df_combined['Time']).dt.date
            df_combined['Tim'] = pd.to_datetime(df_combined['Time']).dt.time
            
            # Calculate busy hour
            if 'GlobelTraffic' in df_combined.columns and 'GCell Group' in df_combined.columns:
                df_bh = df_combined.loc[df_combined.groupby(['Date', 'GCell Group'])['GlobelTraffic'].idxmax()]
                
                # Remove unwanted columns
                df_bh = df_bh.drop(['Date', 'Tim'], axis=1, errors='ignore')
                
                df_bh.to_csv('cluster_bh_output.csv', index=False)
                print("Cluster Busy Hour calculation completed: cluster_bh_output.csv")
            else:
                print("Required columns (GlobelTraffic, GCell Group) not found!")
                
    def process(self, datetime_format="separate"):
        """Main processing function"""
        self.change_working_directory(self.working_directory)
        try:
            if datetime_format == "separate":
                self.process_separate_datetime()
            elif datetime_format == "combined":
                self.process_combined_datetime()
            else:
                print("Invalid datetime format. Use 'separate' or 'combined'")
        finally:
            self.restore_directory()

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="RF Optimization and Planning Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --gsm-worst-cells --input-dir /path/to/gsm/data
  %(prog)s --umts-worst-cells --input-dir /path/to/umts/data
  %(prog)s --hex-to-dec --input-dir /path/to/hex/data --input-file TACLAC.txt
  %(prog)s --cluster-bh --input-dir /path/to/cluster/data --datetime-format separate
  %(prog)s --list-tasks
        """
    )
    
    # Task selection arguments
    parser.add_argument('--gsm-worst-cells', action='store_true',
                       help='Process GSM worst cells from PRS reports')
    parser.add_argument('--umts-worst-cells', action='store_true',
                       help='Process UMTS worst cells from PRS reports')
    parser.add_argument('--hex-to-dec', action='store_true',
                       help='Convert LAC TAC values from hexadecimal to decimal')
    parser.add_argument('--cluster-bh', action='store_true',
                       help='Calculate cluster busy hour from traffic data')
    
    # Configuration arguments
    parser.add_argument('--input-dir', type=str, default='.',
                       help='Input directory path (default: current directory)')
    parser.add_argument('--input-file', type=str, default='TACLAC.txt',
                       help='Input file name for hex-to-dec conversion (default: TACLAC.txt)')
    parser.add_argument('--datetime-format', choices=['separate', 'combined'], default='separate',
                       help='DateTime format for cluster busy hour calculation (default: separate)')
    
    # Utility arguments
    parser.add_argument('--list-tasks', action='store_true',
                       help='List all available tasks')
    parser.add_argument('--version', action='version', version='RF Optimization Tool v1.0')
    
    args = parser.parse_args()
    
    # List available tasks
    if args.list_tasks:
        print("Available Tasks:")
        print("1. --gsm-worst-cells    : Process GSM worst cells from PRS reports")
        print("2. --umts-worst-cells   : Process UMTS worst cells from PRS reports") 
        print("3. --hex-to-dec         : Convert LAC TAC values from hex to decimal")
        print("4. --cluster-bh         : Calculate cluster busy hour from traffic data")
        print("\nUse --help for detailed usage information.")
        return
    
    # Check if at least one task is selected
    tasks_selected = any([args.gsm_worst_cells, args.umts_worst_cells, 
                         args.hex_to_dec, args.cluster_bh])
    
    if not tasks_selected:
        print("No task selected. Use --list-tasks to see available options or --help for usage.")
        return
    
    # Execute selected tasks
    try:
        if args.gsm_worst_cells:
            print("=== GSM Worst Cells Processing ===")
            processor = GSMWorstCellProcessor(args.input_dir)
            processor.process()
            print()
            
        if args.umts_worst_cells:
            print("=== UMTS Worst Cells Processing ===")
            processor = UMTSWorstCellProcessor(args.input_dir)
            processor.process()
            print()
            
        if args.hex_to_dec:
            print("=== Hexadecimal to Decimal Conversion ===")
            processor = HexDecimalConverter(args.input_dir, args.input_file)
            processor.process()
            print()
            
        if args.cluster_bh:
            print("=== Cluster Busy Hour Calculation ===")
            processor = ClusterBusyHourCalculator(args.input_dir)
            processor.process(args.datetime_format)
            print()
            
        print("All selected tasks completed successfully!")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
