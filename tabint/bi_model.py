# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 16:59:43 2017

@author: Yehoshua
github link: https://github.com/yehoshuadimarsky/python-ssas
"""

import pandas as pd
import numpy as np

from functools import wraps
from pathlib import Path
import logging
import warnings

logger = logging.getLogger(__name__)

try:
    import clr  # name for pythonnet
except ImportError:
    msg = """
    Could not import 'clr', install the 'pythonnet' library. 
    For conda, `conda install -c pythonnet pythonnet`
    """
    raise ImportError(msg)

class powerbi:
    def __init__(self, data_source = None, catalog = None, user_name = None, password = None):
        self.data_source, self.catalog, self.user_name, self.password = data_source, catalog, user_name, password
        self.load_assemblies()
        self.connection = f"Provider=MSOLAP;Data Source={self.data_source};Initial Catalog={self.catalog};User ID={self.user_name};Password={self.password};"

    @classmethod
    def from_data_source(cls, data_source): return cls(data_source)
    
    def load_assemblies(self):
        """
        Loads required assemblies, called after function definition.
        Might need to install SSAS client libraries:
        https://docs.microsoft.com/en-us/azure/analysis-services/analysis-services-data-providers

        Parameters
        ----------
        amo_path : str, default None
            The full path to the DLL file of the assembly for AMO. 
            Should end with '**Microsoft.AnalysisServices.Tabular.dll**'
            Example: C:/my/path/to/Microsoft.AnalysisServices.Tabular.dll
            If None, will use the default location on Windows.
        adomd_path : str, default None
            The full path to the DLL file of the assembly for ADOMD. 
            Should end with '**Microsoft.AnalysisServices.AdomdClient.dll**'
            Example: C:/my/path/to/Microsoft.AnalysisServices.AdomdClient.dll
            If None, will use the default location on Windows.
        """

        self.amo_path = "C:/Program Files/PackageManagement/NuGet/Packages/Microsoft.AnalysisServices.retail.amd64.19.6.0/lib/net45/Microsoft.AnalysisServices.Tabular.dll"
        self.adomd_path = "C:/Program Files/PackageManagement/NuGet/Packages/Microsoft.AnalysisServices.AdomdClient.retail.amd64.19.6.0/lib/net45/Microsoft.AnalysisServices.AdomdClient.dll"

        # load .Net assemblies
        logger.info("Loading .Net assemblies...")
        clr.AddReference("System")
        clr.AddReference("System.Data")
        clr.AddReference(self.amo_path)
        clr.AddReference(self.adomd_path)

        # Only after loaded .Net assemblies
        global System, DataTable, AMO, ADOMD

        import System
        from System.Data import DataTable
        import Microsoft.AnalysisServices.Tabular as AMO
        import Microsoft.AnalysisServices.AdomdClient as ADOMD

        logger.info("Successfully loaded these .Net assemblies: ")
        for a in clr.ListAssemblies(True):
            logger.info(a.split(",")[0])


    def _assert_dotnet_loaded(func):
        """
        Wrapper to make sure that required .NET assemblies have been loaded and imported.
        Can pass the keyword arguments 'amo_path' and 'adomd_path' to any annotated function,
        it will use them in the `_load_assemblies` function.

        Example: 
            .. code-block:: python
            
                import ssas_api
                conn = ssas_api.set_conn_string(
                    's', 'd', 'u', 'p', 
                    amo_path='C:/path/number/one', 
                    adomd_path='C:/path/number/two'
                )
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            # amo_path = kwargs.pop("amo_path", None)
            # adomd_path = kwargs.pop("adomd_path", None)
            try:
                type(DataTable)
            except NameError:
                # .NET assemblies not loaded/imported
                logger.warning(".Net assemblies not loaded and imported, doing so now...")
                self.load_assemblies(amo_path=self.amo_path, adomd_path=self.adomd_path)
            return func(*args, **kwargs)
        return wrapper


    @_assert_dotnet_loaded
    def set_conn_string(self, server, db_name, username, password):
        """
        Sets connection string to SSAS database, 
        in this case designed for Azure Analysis Services
        """
        self.conn_string = (
            "Provider=MSOLAP;Data Source={};Initial Catalog={};User ID={};"
            "Password={};Persist Security Info=True;Impersonation Level=Impersonate".format(server, db_name, username, password)
        )

    @_assert_dotnet_loaded
    def get_DAX(self, dax_string):
        """
        Executes DAX query and returns the results as a pandas DataFrame
        
        Parameters
        ---------------
        connection_string : string
            Valid SSAS connection string, use the set_conn_string() method to set
        dax_string : string
            Valid DAX query, beginning with EVALUATE or VAR or DEFINE

        Returns
        ----------------
        pandas DataFrame with the results
        """
        table = self._get_DAX(dax_string)
        df = self._parse_DAX_result(table)
        return df


    def _get_DAX(self, dax_string) -> "DataTable":
        dataadapter = ADOMD.AdomdDataAdapter(dax_string, self.connection)
        table = DataTable()
        logger.info("Getting DAX query...")
        dataadapter.Fill(table)
        logger.info("DAX query successfully retrieved")
        return table


    def _parse_DAX_result(self, table: "DataTable") -> pd.DataFrame:
        cols = [c for c in table.Columns.List]
        rows = []
        # much better performance to just access data by position instead of name
        # and then add column names afterwards
        for r in range(table.Rows.Count):
            row = [table.Rows[r][c] for c in cols]
            rows.append(row)

        df = pd.DataFrame.from_records(rows, columns=[c.ColumnName for c in cols])

        # replace System.DBNull with None
        # df.replace({System.DBNull: np.NaN}) doesn't work for some reason
        df = df.applymap(lambda x: np.NaN if isinstance(x, System.DBNull) else x)

        # convert datetimes
        dt_types = [c.ColumnName for c in cols if c.DataType.FullName == "System.DateTime"]
        if dt_types:
            for dtt in dt_types:
                # if all nulls, then pd.to_datetime will fail
                if not df.loc[:, dtt].isna().all():
                    ser = df.loc[:, dtt].map(lambda x: x.ToString())
                    df.loc[:, dtt] = pd.to_datetime(ser)

        # convert other types
        types_map = {"System.Int64": int, "System.Double": float, "System.String": str}
        col_types = {c.ColumnName: types_map.get(c.DataType.FullName, "object") for c in cols}
        
        # handle NaNs (which are floats, as of pandas v.0.25.3) in int columns
        col_types_ints = {k for k,v in col_types.items() if v == int}
        ser = df.isna().any(axis=0)
        col_types.update({k:float for k in set(ser[ser].index).intersection(col_types_ints)})
        
        # convert
        df = df.astype(col_types)

        return df


    @_assert_dotnet_loaded
    def process_database(self, refresh_type, db_name):
        self.process_model(
            refresh_type=refresh_type,
            item_type="model"

        )


    @_assert_dotnet_loaded
    def process_table(self, table_name, refresh_type, db_name):
        self.process_model(
            refresh_type=refresh_type,
            item_type="table",
            item=table_name
        )

    @_assert_dotnet_loaded
    def process_model(self, refresh_type="full", item_type="model", item=None):
        """
        Processes SSAS data model to get new data from underlying source.
        
        Parameters
        -------------
        connection_string : string
            Valid SSAS connection string, use the set_conn_string() method to set
        db_name : string
            The data model on the SSAS server to process
        refresh_type : string, default `full`
            Type of refresh to process. Currently only supports `full`.
        item_type : string, choice of {'model','table'}, default 'model'
        item : string, optional.
            Then name of the item. Only needed when item_type is 'table', to specify the table name
        """
        assert item_type.lower() in ("table", "model"), f"Invalid item type: {item_type}"
        if item_type.lower() == "table" and not item:
            raise ValueError("If item_type is table, must supply an item (a table name) to process")

        # connect to the AS instance from Python
        AMOServer = AMO.Server()
        logger.info("Connecting to database...")
        AMOServer.Connect(self.conn_string)

        # Dict of refresh types
        refresh_dict = {"full": AMO.RefreshType.Full}

        # process
        db = AMOServer.Databases[self.catalog]

        if item_type.lower() == "table":
            table = db.Model.Tables.Find(item)
            table.RequestRefresh(refresh_dict[refresh_type])
        else:
            db.Model.RequestRefresh(refresh_dict[refresh_type])

        op_result = db.Model.SaveChanges()
        if op_result.Impact.IsEmpty:
            logger.info("No objects affected by the refresh")

        logger.info("Disconnecting from Database...")
        # Disconnect
        AMOServer.Disconnect()