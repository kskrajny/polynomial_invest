"""data_downloader package
1. Basic info
A simple package to download data from Oanda. The main aim is to emphasize the simplicity of use. Currently supports
all OHLC instruments and ["COT," "FRED"] data. Current database queries don't query future margins and past dates as in
the framework. Package contains utils file, which is currently only for developer use.
2. Installation
a) Install using wheel file -> pip install {package_name}
b) Install package in editable mode with pip:
    git pull development
    cd {root_path}/ait-ng/src/data_downloader_package/
    pip install -e .
3. Supported functionalities
DataDownloader __init__: A class for downloading the data from DB
    :param username: Username to input into db_connection string
    :param password: Password to input into db_connection string
    :param db_address: An address of a IP defaults to: '116.203.105.121:3307'
get_dataframes: The main function to download data frames from database
    :param instruments: A list of strings with names of instruments for which data should be downloaded
    :param date_from: Date from which download the data
    :param date_to: Date to which download the data.
    :return: A list of data frames with requested data, one for each instrument
get_single_dataframe: The main function to get single wide data frame with all columns from each instrument selected
    :param instruments: A list of strings with names of instruments for which data should be downloaded
    :param date_from: Date from which download the data
    :param date_to: Date to which download the data.
    :return: A single data frame in which each column has a suffix with an instrument name. Since some instruments
    might have no data at a given day, NaN values in data are expected.
"""
import logging
from pprint import pformat
from typing import List, Dict, Tuple
from datetime import datetime

import pandas as pd

from data_downloader.utils import get_default_logger, SQLSessionConnector


class DataDownloader:

    def __repr__(self):
        return f"This DataDownloader has following attributes: {self.__dict__}"

    def __str__(self):
        return f"This DataDownloader has following attributes: \n {pformat(self.__dict__)}"

    def __del__(self):
        if self._session is not None:
            self._session.close()

    def __init__(self,
                 username: str,
                 password: str,
                 db_address: str = "192.168.11.128:3308",
                 provider: str = 'OANDA'):
        """A class for downloading data from db
        :param username: Username to input into db_connection string
        :param password: Password to input into db_connection string
        :param db_address: An address of a ip e.g. '116.203.105.121:3307'
        :param provider #OANDA IC_MARKETS
        """
        # Setup logger, minor instruments; handle minor instruments separately
        self.logger = get_default_logger()
        self.minor_instruments = ["COT", "FRED"]  # "FOREX_CALENDAR"

        # DB connection and setup session
        self._process_name = "data_downloader"
        self._db_connection_string = f"mysql://{username}:{password}@{db_address}/ait"
        self._session_connector = SQLSessionConnector()
        self._session = self._session_connector.sql_session(self._db_connection_string,
                                                            self._process_name)

        # Setup column names and index values for SQL queries
        self._instrument_map = {
            "OHLC": {"index": "timestamp", "cols": ["timestamp", "open", "high", "low", "close", "dayofweek","volume"]},
            "FOREX_CALENDAR": {"index": "datetime",
                               "cols": ["datetime", "time", "currency", "impact", "event", "actual",
                                        "forecast", "previous", "actual_diff", "previous_diff"]},
            "COT": {"index": "Timestamp",
                    "cols": ["Timestamp",
                             "Dealer_Long", "Dealer_Short", "Dealer_Spreading",
                             "Asset_Manager_Long", "Asset_Manager_Short", "Asset_Manager_Spreading",
                             "Leveraged_Funds_Long", "Leveraged_Funds_Short", "Leveraged_Funds_Spreading",
                             "Other_Reportables_Long", "Other_Reportables_Short", "Other_Reportables_Spreading",
                             "Nonreportable_Positions_Long", "Nonreportable_Positions_Short"]},
            "FRED": {"index": "Timestamp",
                     "cols": ["Timestamp", "Value"]}}
        
        self.provider=provider

    def get_dataframes(self,
                       instruments: List[str],
                       date_from: datetime,
                       date_to: datetime) -> Dict[str, pd.DataFrame]:
        """Main function to download dataframes from database
        :param instruments: A list of strings with names of instruments for which data should be downloaded
        :param date_from: Date from which download the data
        :param date_to: Date to which download the data.
        :return: A list of dataframes with requested data, one for each instrument
        """
        dataframes = {}

        for instrument in instruments:
            # Setup str formatting for SQL queries. Extract instrument type from it's whole name
            format_dict = dict(date_from=date_from, date_to=date_to, instrument=instrument,provider=self.provider)
            instrument_name = instrument.split("_")[0]

            # Decide on a type of query (major or minor instruments)
            if instrument_name not in self.minor_instruments:
                results_dict = self._prepare_ohlc_query(format_dict=format_dict)
            else:
                results_dict = self._prepare_minor_instrument_query(format_dict=format_dict,
                                                                    instrument_name=instrument_name)

            # Format query, run it, preprocess it and add to dfs list
            query_f = results_dict["query"]().format(**format_dict)
            query_results = self._invoke_sql_query(query_f)
            dataframe = DataDownloader._preprocess_dfs(data=query_results,
                                                       columns=results_dict["single_instrument_map"],
                                                       datetime_index=results_dict["single_instrument_index"])
            dataframes[instrument] = dataframe

        return dataframes

    def get_single_dataframe(self,
                             instruments: List[str],
                             date_from: datetime,
                             date_to: datetime) -> pd.DataFrame:
        """Main function to get single wide dataframe with all columns from each instrument selected
        :param instruments: A list of strings with names of instruments for which data should be downloaded
        :param date_from: Date from which download the data
        :param date_to: Date to which download the data.
        :return: A single dataframe in which each column has a suffix with an instrument name. Since some instruments
        might have no data at given day, NaN values in data are expected.
        """
        dataframes_df = self.get_dataframes(instruments, date_from, date_to)

        # Prepare df with all indexes and days of week available
        all_index_df_input = []
        for df in dataframes_df.values():
            try:
                all_index_df_input.append(df[["day_of_week"]])
            except KeyError:
                all_index_df_input.append(pd.DataFrame([], index=df.index))

        # Remove duplicated indexes
        all_index_df = pd.concat(all_index_df_input)
        duplicated_indexes = all_index_df.index.duplicated(keep="first")
        all_index_df = all_index_df[~duplicated_indexes]

        # Add suffix, join dfs and remove redundant dayofweek and timestamp columns
        for key, val in dataframes_df.items():
            # Handle different names of timestamp column
            val = val.drop(([val["index"] for val in self._instrument_map.values()]
                            + ["dayofweek"]),
                           axis=1,
                           errors="ignore")
            # val = val.add_suffix(f"_{key}")
            all_index_df = all_index_df.join(val, how="left")
        return all_index_df

    def _close(self):
        """Utility function to invalidate current sql session after deletion of an object"""
        self._session_connector.invalidate_session(self._db_connection_string, self._process_name)
        self.__del__()

    def _invoke_sql_query(self, sql_query: str) -> List[Tuple]:
        """Run SQL query and handle it's errors
        :param sql_query: A formatable sql query to be run
        :return: SQL query results in a format of list of tuples
        """
        self.logger.debug("invoking sql query: %s" % sql_query)
        try:
            result = self._session.execute(sql_query)
        except Exception as e:
            self.logger.error(e, exc_info=True)
            logging.exception(">>>>>>> load_data_from_db <<<<<<<<<")
            try:
                self._session = self._session_connector.sql_session(self._db_connection_string,
                                                                    self._process_name)
                return self._invoke_sql_query(sql_query)
            except Exception:
                logging.exception("load_data_from_db")
                result = None
        return result.fetchall()

    @staticmethod
    def _ohlc_query() -> str:
        """SQL Query to get OHLC from OANDA broker
        :return: Formatable string query to DB with following parameters: instrument, date_from, date_to,\
         margin, range.
        """
        
        return  "( " \
               "SELECT barTimestamp AS DATE, OPEN, high, low, CLOSE, dayofweek, volume " \
               "FROM aitbars.{instrument} " \
               "WHERE (barTimestamp >= '{date_from}' AND barTimestamp <= '{date_to}') AND provider = '{provider}' " \
               "ORDER BY barTimestamp" \
               ")"

    @staticmethod
    def _minor_instrument_query() -> str:
        """SQL Query to get xxx from db
        """
        return "( " \
               "SELECT {columns_str} " \
               "FROM aitbars.{instrument} " \
               "WHERE ({date_col} >= '{date_from}' AND {date_col} <= '{date_to}')  " \
               "ORDER BY {date_col} " \
               ") "

    def _prepare_ohlc_query(self, format_dict: dict) -> dict:
        """Utility function to prepare a dictionary with several variables for later use in SQL query
        :param format_dict: A dictionary with formatting. NOTE: Function is making changes inplace in used dict
        :return: Returns a dict with single instrument map, idex and a proper SQL query
        """
        # Check if period available, add columns and index column to query, choose appropriate query

        single_instrument_map = self._instrument_map["OHLC"]["cols"]
        format_dict["columns"] = single_instrument_map

        single_instrument_index = self._instrument_map["OHLC"]["index"]

        query = DataDownloader._ohlc_query
        results_dict = {"single_instrument_map": single_instrument_map,
                        "single_instrument_index": single_instrument_index,
                        "query": query}
        return results_dict

    def _prepare_minor_instrument_query(self, format_dict: dict, instrument_name: str) -> dict:
        """Utility function to prepare a dictionary with several variables for later use in SQL query
        :param format_dict: A dictionary with formatting. NOTE: Function is making changes inplace in used dict
        :param instrument_name: A type of instrument to use in query e.g. "COT", "FRED"
        :return: Returns a dict with single instrument map, index and a proper SQL query
        """
        # Filter instrument name from self._instrument_map
        single_instrument = next(v for k, v
                                 in self._instrument_map.items()
                                 if k.startswith(instrument_name))  # Prone to error for ins with similar name

        # Set columns in list and string formats for given instrument, set index, choose query
        single_instrument_map = single_instrument["cols"]
        format_dict["columns"] = single_instrument_map
        format_dict["columns_str"] = ", ".join(single_instrument_map)

        single_instrument_index = single_instrument["index"]
        format_dict["date_col"] = single_instrument_index

        query = DataDownloader._minor_instrument_query
        results_dict = {"single_instrument_map": single_instrument_map,
                        "single_instrument_index": single_instrument_index,
                        "query": query}
        return results_dict

    @staticmethod
    def _preprocess_dfs(data: List[tuple], columns: List[str], datetime_index: str) -> pd.DataFrame:
        """Preprocess tuple to dataframe, handle columns and indexes for multiple results
        :param data: Data in a format of list of tuples
        :param columns: Column names list to be set from a SQL query
        :param datetime_index: Name of a datetime column to be chosen as an index
        :return: A preprocessed DataFrame
        """
        dataframe = pd.DataFrame(data=data)
        if len(data) > 0:
            dataframe.columns = columns
            dataframe = dataframe.set_index(pd.DatetimeIndex(dataframe[datetime_index]))
        return dataframe
