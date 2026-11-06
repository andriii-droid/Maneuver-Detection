"""Preprocess raw data files into feature dataframes."""

import pandas as pd
from tqdm import tqdm
import numpy as np

from maneuver_detection.config import Config


class Preprocessor:
    """Preprocess raw data files into feature dataframes."""

    def __init__(self, config: Config) -> None:
        """Initialize.

        Args:
            config: Project configuration.
        """
        self._config = config
        state_descriptions = pd.read_json(self._config.state_description_path)
        self._state_id_to_name_mapper = {
            value: column for column, value in state_descriptions.loc["stateId"].dropna().items()
        }

    def preprocess(self, data_ids: list[str]) -> pd.DataFrame:
        """Preprocess multiple files and concatenate the results."""
        df_preprocessed = pd.DataFrame()
        pd.set_option("display.max_rows", 100)

        for data_id in tqdm(data_ids):
            df = self.preprocess_file(data_id)
            df.columns = df.columns.str.strip()

            df_preprocessed = pd.concat([df_preprocessed, df], ignore_index=True)
        
        return df_preprocessed.fillna(0)

    def preprocess_file(self, data_id: str) -> pd.DataFrame:
        """Preprocess a single file identified by data_id."""
        annotations = self._read_annotations(data_id)
        df = self._read_parquet_file(data_id)
        df = self._rename_columns(df)
        df = self._extract_relevant_columns(df) #TODO
        df = _convert_dtypes(df)
        df = self._fix_missing_values(df) #done
        df = _fix_index(df)
        df = _expand_vectors(df)
        df = self._assign_maneuver_labels(df, annotations) #done
        df = self._summarize_features_in_blocks(df)
        return df

    def create_train_set(self) -> pd.DataFrame:
        """Preprocess the training dataset."""
        df = self.preprocess(self._config.train_data_ids)
        df.to_parquet(self._config.train_data_path)
        return df

    def create_test_set(self) -> pd.DataFrame:
        """Preprocess the testing dataset."""
        df = self.preprocess(self._config.test_data_ids)
        df.to_parquet(self._config.test_data_path)
        return df

    def _read_parquet_file(self, data_id: str) -> pd.DataFrame:
        return pd.read_parquet(self._config.data_path / f"{data_id}.parquet")

    def _read_annotations(self, data_id: str) -> pd.DataFrame:
        return pd.read_json(self._config.data_path / f"{data_id}.json")["annotations"]

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.rename(columns=self._state_id_to_name_mapper)

    def _extract_relevant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO: Implement relevant column extraction
        return df

    def _fix_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        '''Fills missing Values with the value of t-1'''
        return df.ffill().bfill()


    def _assign_maneuver_labels(self, df: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
        ann_df = pd.DataFrame(annotations.tolist())

        labels = {str(k).lower(): v for k, v in self._config.label2id.items()}

        ann_df["startTimestamp"] = pd.to_datetime(ann_df["startTimestamp"])
        ann_df["endTimeStamp"] = pd.to_datetime(ann_df["endTimeStamp"])
        ann_df["comment"] = ann_df["comment"].str.lower().map(labels)
        ann_df = ann_df.sort_values("startTimestamp")
 
        df["TimeStamp"] = pd.to_datetime(df["TimeStamp"]).astype("datetime64[us]")
        df = df.sort_values("TimeStamp")

        merged = pd.merge_asof(
            df,
            ann_df[["startTimestamp", "endTimeStamp", "comment"]],
            left_on="TimeStamp",
            right_on="startTimestamp",
            direction="backward",
        )

        merged["Maneuver_Label"] = merged["comment"].fillna(0)

        merged = merged.drop(columns=["startTimestamp", "endTimeStamp", "comment"])
        return merged

    def _summarize_features_in_blocks(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.set_index("TimeStamp")
        time = df.index[0]
        end_time = df.index[-1]
        df_features = pd.DataFrame()
        block_size = pd.Timedelta(seconds=self._config.preprocess.block_size)
        stride_size = pd.Timedelta(seconds=self._config.preprocess.stride_size)

        while time <= (end_time - block_size):

            block = df.loc[(df.index >= time) & (df.index < time + block_size)]
            
            block_mean = block.select_dtypes(include="number").mean().to_frame().T
            block_mean = block_mean.rename(columns=lambda x: f"{x}_Mean")

            block_std = block.select_dtypes(include="number").std().to_frame().T
            block_std = block_std.rename(columns=lambda x: f"{x}_Std")

            block_mode = block.select_dtypes(include=["boolean", "bool"]).mode()
            block_mode = block_mode.rename(columns=lambda x: f"{x}_Mode")

            # TODO: Add more or change features

            block_concat = pd.concat([block_mean, block_std, block_mode], axis="columns")

            modes = block["Maneuver_Label"].mode()
            block_concat["Maneuver_Label"] = modes.iloc[0] if not modes.empty else None

            df_features = pd.concat([df_features, block_concat], ignore_index=True)

            time += stride_size



        #df_features = df_features.dropna(ignore_index=True)
        
        if df_features.isna().any().any():
            print("Found NaN in DataFrame!")
        return df_features

def _fix_index(df: pd.DataFrame) -> pd.DataFrame:
    # Drop FrameCounter level
    index = df.index.droplevel("FrameCounter")

    # Remove timezone information
    index = pd.to_datetime(index).tz_localize(None)
    df.index = index
    return df.reset_index()


def _convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    return df.convert_dtypes()


def _expand_vectors(df: pd.DataFrame) -> pd.DataFrame:
    for column in df.columns:
        if hasattr(df[column].iloc[0], "__len__"):
            vector_length = len(df[column].iloc[0])
            dtype = df[column].dtype
            if vector_length > 1 and dtype != "string":
                expanded = pd.DataFrame(
                    df[column].tolist(),
                    columns=[f"{column}_{i}" for i in range(vector_length)],
                    index=df.index,
                )

                df = pd.concat([df, expanded], axis="columns")
                df = df.drop(columns=[column])
    return df


if __name__ == "__main__":
    from maneuver_detection.config import get_default_config

    _config = get_default_config()
    preprocessor = Preprocessor(_config)

    print(f"Preprocessing training data and writing results to {_config.train_data_path}")
    preprocessor.create_train_set()

    # print(f"Preprocessing test data and writing results to {_config.test_data_path}")
    # preprocessor.create_test_set()
