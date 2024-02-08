import numpy as np
import pandas as pd


def get_bin_columns(dataframe):
    columnas_binarias = []
    columnas_no_binarias = []

    for columna in dataframe.columns:
        valores_unicos = dataframe[columna].dropna().unique()

        if len(valores_unicos) == 2:
            columnas_binarias.append(columna)
        else:
            columnas_no_binarias.append(columna)

    return columnas_binarias, columnas_no_binarias


def fill_with_encoding(dataframe: pd.DataFrame, columns: list) -> pd.DataFrame:
    # Reemplazar los 0 por -1
    dataframe[columns] = dataframe[columns].replace(0, -1)

    dataframe[columns] = dataframe[columns].fillna(0)

    return dataframe[columns]


def fill_with_mean(dataframe: pd.DataFrame, columns: list):
    for column in columns:
        mean = dataframe[column].mean()
        dataframe[column].fillna(mean, inplace=True)

    return dataframe[columns]


def fast_fill(dataframe: pd.DataFrame):
    # binary_cols, non_binary_cols = get_bin_columns(dataframe)
    binary_cols = [
        "pahip50e",
        "pahibpe",
        "parespe",
        "pacholst",
        "pagrossaa",
        "pafallinj",
        "pacancre",
        "paswell",
        "parafaany",
        "pawheeze",
        "paarthre",
        "padiabe",
        "pastroke",
        "pasmokev",
        "pahearaid",
        "paglasses",
        "papaina",
        "pagender",
        "padrinkb",
    ]
    non_binary_cols = [
        "padadage",
        "pamomage",
        "paosleep",
        "paagey",
        "parjudg",
        "paweight",
        "paclims",
        "pameds",
        "paheight",
        "paeat",
    ]
    dataframe[binary_cols] = fill_with_encoding(dataframe, binary_cols)
    dataframe[non_binary_cols] = fill_with_mean(dataframe, non_binary_cols)

    return dataframe


def get_corr_columns(dataframe: pd.DataFrame, column: str, x: bool = False):
    correlations = dataframe.corr()[column].abs().sort_values(ascending=x)
    correlated_columns = correlations.index[1:25].tolist()

    return correlated_columns
