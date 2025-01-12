import pandas as pd
import os
import glob
from datetime import datetime

# Global Variables
ORIGINAL_FILE = 'SampleReport_181_allvars_14jan2023.txt'
PIDS_FOLDER = 'SampleReport_PIDS/'
COMPLETED_FILE_FOLDER = 'Downsampled_Files/'

class DataWrangling:
    '''
    Takes 78 GB raw Dataviewer data, divides up into separate file for each pacritipant,
    and finally downsamples them to disired sampling rate/
    
    '''
    downsampleHZ = 100 #HZ value to downsample too. Origionally at 1000 HZ.
                       #For isntance, downsampling to 100 HZ means choosing every 1000/100 = 10th row

    def __init__(self):
        self.now = datetime.now()

    @property
    def downsampled_pids_folder(self):
        return f"{DataWrangling.downsampleHZ}HZ_PIDS/"

    @staticmethod
    def parse_identifier(df):
        """
        Parses the identifier column to extract relevant information and create new columns.
        """
        df["ParticipantID"] = df.RECORDING_SESSION_LABEL.str[0:8]
        df['FilePart'] = df.RECORDING_SESSION_LABEL.str[-5:]
        df["PartNo"] = df.RECORDING_SESSION_LABEL.str[-1:]

        # Define trial type conditions
        def trialtype_conditions(row):
            if 'Sham' in row['identifier']: return 'sham'
            if 'Practice' in row['identifier']: return 'Prac'
            if 'MW' in row['identifier']: return 'MW'
            if 'SVT' in row['identifier']: return 'SVT'
            if 'Inf' in row['identifier']: return 'Inference'
            if 'Rote' in row['identifier']: return 'Rote'
            if 'Deep' in row['identifier']: return 'Deep'
            if 'DriftCorrect' in row['identifier']: return 'DriftCorrect'
            if 'Recal' in row['identifier']: return 'Recal'
            if 'BIGBREAK' in row['identifier']: return 'Break'
            if 'Resting' in row['identifier']: return 'restingState'
            if 'Localizer' in row['identifier']: return 'Localizer'
            if 'IBI began' in row['identifier']: return 'Localizer'
            if 'Lang Task' in row['identifier']: return 'Localizer'
            if 'UNDEFINED' in row['identifier']: return 'UNDEFINED'
            if ('EML1_001' in row["ParticipantID"]) and ('sham' in row['identifier'].lower()): return "NA"
            if 'ShamStart' in row['identifier']: return "NA"
            return 'reading'

        df["TrialType"] = df.apply(trialtype_conditions, axis=1)

        # Define text conditions
        def text_conditions(row):
            if 'Validity' in row['identifier']: return 'Validity'
            if 'Bias' in row['identifier']: return 'Bias'
            if 'CausalClaims' in row['identifier']: return 'CausalClaims'
            if 'Variables' in row['identifier']: return 'Variables'
            if 'Hypotheses' in row['identifier']: return 'Hypotheses'
            return "NA"

        df["Text"] = df.apply(text_conditions, axis=1)

        # Define stage conditions
        def stage_conditions(row):
            if 'Z_Question' in row['identifier']: return 'Y'
            if 'Y_Question' in row['identifier']: return 'Z'
            return "X"

        df["Stage"] = df.apply(stage_conditions, axis=1)

        # Define page number conditions
        def pagenum_conditions(row):
            for s in row.identifier:
                if s.isdigit():
                    p = int(s)
                    if row['TrialType'] in ['reading', 'sham', 'prac']:
                        p += 1
                    return p
            return ""

        df['PageNum'] = df.apply(pagenum_conditions, axis=1)

        return df

    def create_pid_samplereports(self, output_folder):
        """
        Splits the original file into separate files for each participant and saves them to a folder.
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        chunk_size = 10**6
        for chunk in pd.read_csv(ORIGINAL_FILE, sep='\t', chunksize=chunk_size, low_memory=False):
            chunk = self.parse_identifier(chunk)
            for participant_id, participant_data in chunk.groupby('ParticipantID'):
                output_file = os.path.join(output_folder, f"{participant_id}.csv")
                if os.path.exists(output_file):
                    participant_data.to_csv(output_file, mode='a', index=False, header=False)
                else:
                    participant_data.to_csv(output_file, index=False)

    @staticmethod
    def downsample(df):
        """
        Downsamples the DataFrame based on the specified downsample rate.
        """
        downsample_rows = 1000/DataWrangling.downsampleHZ
        return df.iloc[::downsample_rows, :]

    def downsample_perpid(self):
        """
        Reads participant files, downsamples them, and saves the results.
        """
        output_folder = self.downsampled_pids_folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for file in glob.glob(os.path.join(PIDS_FOLDER, "*.csv")):
            file_name = os.path.basename(file)
            df = pd.read_csv(file, low_memory=False)
            df_downsampled = df.groupby(['ParticipantID', 'identifier']).apply(self.downsample)
            output_file = os.path.join(output_folder, file_name)
            df_downsampled.to_csv(output_file, index=False)

    def concat_pids(self, input_folder, output_file):
        """
        Concatenates all downsampled participant files into one file.
        """
        cols = [
            "RECORDING_SESSION_LABEL", "identifier", "ParticipantID", "Text", "PageNum", "TrialType", 
            "TRIAL_INDEX", "AVERAGE_GAZE_X", "AVERAGE_GAZE_Y", "TIMESTAMP", "TRIAL_START_TIME",
            "AVERAGE_VELOCITY_X", "AVERAGE_VELOCITY_Y", "AVERAGE_ACCELERATION_X", "AVERAGE_ACCELERATION_Y"
        ]

        data_frames = []

        for file_name in os.listdir(input_folder):
            file_path = os.path.join(input_folder, file_name)
            if os.path.isfile(file_path) and file_name.endswith('.csv'):
                try:
                    df = pd.read_csv(file_path, low_memory=False)
                    data_frames.append(df[cols])
                except Exception as e:
                    print(f"Error processing file {file_name}: {e}")

        if data_frames:
            final_df = pd.concat(data_frames, ignore_index=True)
            final_df.to_csv(output_file, index=False)
            print(f"Concatenated file saved as {output_file}")
        else:
            print("No files to concatenate.")

if __name__ == "__main__":
    dw = DataWrangling()
    dw.create_pid_samplereports(PIDS_FOLDER)
    dw.downsample_perpid()
    dw.concat_pids(dw.downsampled_pids_folder, os.path.join(COMPLETED_FILE_FOLDER, f"Gaze_{DataWrangling.downsampleHZ}hz.csv"))