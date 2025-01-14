import pandas as pd
import numpy as np
import glob
import math
from typing import Tuple, List


class GazeDataLoader:
    """
    A class for loading and processing eye-tracking data from Dataviewer.
    """

    def __init__(self, data_path: str):
        """
        Initializes the loader with the path where eye-tracking data is stored.
        
        Args:
            data_path (str): Path to the directory containing event files.
        """
        self.data_path = data_path

    def load_events(self, subject: str, trial_type: str = "reading") -> pd.DataFrame:
        """
        Loads event data for a given subject and filters by trial type.

        Args:
            subject (str): Subject identifier.
            trial_type (str): Type of trial to filter (default: "reading").

        Returns:
            pd.DataFrame: Filtered events data.
        """
        events = pd.read_csv(f"{self.data_path}/{subject}_events.csv")
        events['Text'] = events['Text'].fillna(method='bfill')

        events['TrialType'] = events['VAL'].map({7: 'reading', 20: 'sham'})
        events = events.loc[events['TrialType'] == trial_type].reset_index(drop=True)

        if 'eye_sample' in events.columns:
            events = events.dropna(subset=['eye_sample']).reset_index(drop=True)

        return events

    def load_sham(self, subject: str) -> pd.DataFrame:
        """
        Extracts sham pages and corresponding reading pages to create a sham dataset.

        Args:
            subject (str): Subject identifier.

        Returns:
            pd.DataFrame: Sham event data.
        """
        events = self.load_events(subject, trial_type="sham")
        sham_texts = events['Text'].tolist()
        sham_pages = events['PageNum'].tolist()

        sham_events = pd.concat(
            [events[(events.PageNum == page) & (events.Text == text)] 
             for page, text in zip(sham_pages, sham_texts)]
        ).reset_index(drop=True)

        return sham_events

    @staticmethod
    def find_nearest(array: np.ndarray, value: float) -> Tuple[float, int]:
        """
        Finds the nearest value in an array and its index.

        Args:
            array (np.ndarray): Sorted array to search.
            value (float): Value to find.

        Returns:
            Tuple[float, int]: (Nearest value, Index)
        """
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or abs(value - array[idx - 1]) < abs(value - array[idx])):
            return array[idx - 1], idx - 1
        return array[idx], idx

    def parse_events(self, events: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """
        Tags events in the provided dataset based on time stamps.

        Args:
            events (pd.DataFrame): Event data containing 'eye_sample' and duration info.
            data (pd.DataFrame): Data to tag with events.

        Returns:
            pd.DataFrame: Data with tagged events.
        """
        data['event'] = np.nan
        for _, event in events.iterrows():
            start_time, start_idx = self.find_nearest(data['tStart'].values, event['eye_sample'])
            end_time, end_idx = self.find_nearest(data['tEnd'].values, start_time + event['duration_sec'] * 1000)
            data.loc[start_idx+1:end_idx, 'event'] = event['EVENT']

        return data.dropna(subset=['event']).reset_index(drop=True)

    def parse_blinks(self, events: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts blinks from event and sample data.

        Args:
            events (pd.DataFrame): Event data.
            data (pd.DataFrame): Eye-tracking sample data.

        Returns:
            pd.DataFrame: Data with identified blink events.
        """
        return self.parse_events(events, data)

    def parse_sample_data(self, events: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """
        Parses sample data using event timing.

        Args:
            events (pd.DataFrame): Event data.
            data (pd.DataFrame): Sample data.

        Returns:
            pd.DataFrame: Data with tagged sample events.
        """
        return self.parse_events(events, data)

    def load_gaze_files(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads and processes gaze data including fixations, saccades, blinks, and samples.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
            (Fixations, Saccades, Blinks, Samples)
        """
        fix_files = glob.glob(f"{self.data_path}/*_Fixation.csv")
        sac_files = glob.glob(f"{self.data_path}/*_Saccade.csv")
        blink_files = glob.glob(f"{self.data_path}/*_Blink.csv")
        sample_files = glob.glob(f"{self.data_path}/*_Sample.csv")

        def load_and_filter(files: List[str], event_type: str) -> pd.DataFrame:
            all_data = pd.concat(
                [pd.read_csv(file).query("eye == 'R'") if 'eye' in pd.read_csv(file).columns else pd.read_csv(file)
                 for file in files], ignore_index=True)
            return all_data.sort_values(by='tStart' if event_type != 'Sample' else 'tSample').drop_duplicates().reset_index(drop=True)

        all_fix = load_and_filter(fix_files, "Fixation")
        all_sac = load_and_filter(sac_files, "Saccade")
        all_blink = load_and_filter(blink_files, "Blink")
        all_sample = load_and_filter(sample_files, "Sample")

        # Interpolate missing pupil size
        all_sample['pupil_size'] = all_sample[['RPupil', 'LPupil']].apply(lambda row: row['RPupil'] if row['RPupil'] < row['LPupil'] else row['LPupil'], axis=1)
        all_sample['pupil_size'].replace(0, np.nan, inplace=True)
        all_sample['pupil_size'] = all_sample['pupil_size'].interpolate(method='linear', limit_direction='both')

        # Clean outliers
        all_sac = all_sac[(all_sac.ampDeg > 0) & (5 < all_sac.vPeak < 1000) & (all_sac.ampDeg < 20) & (all_sac.duration < 600)]
        all_fix = all_fix[(40 < all_fix.duration) & (all_fix.duration < 1000)]

        return all_fix, all_sac, all_blink, all_sample



