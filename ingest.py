import pandas as pd
import os 
from tqdm import tqdm
import astropy.units as u
import matplotlib.pyplot as plt

# not reccomended to suppress any warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def ingest_data(file_path, extension='parquet', columns=['diaObject', 'prvDiaSources', 'prvDiaForcedSources', 'lc_features', 'xm']):
    """Ingest data from FINK data transfer of Rubin alerts. Returns stitched pd.DataFrame
    of downloaded alerts. Assumes a `Medium packet` content.

    Parameters:
    ----------
    - file_path: str, path to the directory containing the data files
    - extension: str, file extension to look for (default: 'parquet')
    - columns: list, columns to include in the final DataFrame (default: 'diaObject', 'prvDiaSources', 'prvDiaForcedSources', 'lc_features', 'xm').

    Returns:
    -------
    pd.DataFrame: A DataFrame containing the combined ingested data.

    Note: 
    -----
    See https://lsst.fink-portal.org/schemas for more information.
    """
    # generate all paths
    parquet_files = [os.path.join(file_path, file) for file in os.listdir(file_path) if file.endswith(f'.{extension}')]

    if not parquet_files:
        raise FileNotFoundError(f"No files with extension '{extension}' found in {file_path}.")

    if columns=='all':
        tables = [pd.read_parquet(file) for file in tqdm(parquet_files)]
    else:
        tables = [pd.read_parquet(file)[columns] for file in tqdm(parquet_files)]

    combined = pd.concat(tables) # stitch all tables

    # check if combined has diaObjectId; if not add it
    if 'diaObjectId' not in combined.columns:
        combined['diaObjectId'] = [val['diaObjectId'] for val in combined['diaObject']]
    
    return combined

def unpack_lc_features(table, diaObjectId):
    """Unpack the column lc_features and return a stitched dataframe for each band given a diaObjectId.
    
        Parameters
        ----------
        table: pd.DataFrame
            The input table containing light curve features.
        diaObjectId: int
            The diaObjectId for which to unpack the light curve features.
    """

    select_table = table[table['diaObjectId'] == diaObjectId]

    feature_table_all = pd.DataFrame(columns=['band', 
                        'amplitude',
                        'anderson_darling_normal',
                        'beyond_1_std',
                        'chi2',
                        'cusum',
                        'inter_percentile_range_10',
                        'kurtosis',
                        'linear_fit_reduced_chi2',
                        'linear_fit_slope',
                        'linear_fit_slope_sigma',
                        'linear_trend',
                        'linear_trend_noise',
                        'linear_trend_sigma',
                        'magnitude_percentage_ratio_20_10',
                        'magnitude_percentage_ratio_40_5',
                        'maximum_slope',
                        'mean',
                        'mean_variance',
                        'median',
                        'median_absolute_deviation',
                        'median_buffer_range_percentage_10',
                        'percent_amplitude',
                        'skew',
                        'standard_deviation',
                        'stetson_K',
                        'weighted_mean'])
    
    for i, bband in enumerate(list('ugrizy')):
        select_filter_table = select_table['lc_features'].iloc[0][f'{bband}']
        if select_filter_table:
            feature_table_all.loc[i] = {
                "band": bband,
                "amplitude": select_filter_table['amplitude'],
                "anderson_darling_normal": select_filter_table['anderson_darling_normal'],
                "beyond_1_std": select_filter_table['beyond_1_std'],
                "chi2": select_filter_table['chi2'],
                "cusum": select_filter_table['cusum'],
                "inter_percentile_range_10": select_filter_table['inter_percentile_range_10'],
                "kurtosis": select_filter_table['kurtosis'],
                "linear_fit_reduced_chi2": select_filter_table['linear_fit_reduced_chi2'],
                "linear_fit_slope": select_filter_table['linear_fit_slope'],
                "linear_fit_slope_sigma": select_filter_table['linear_fit_slope_sigma'],
                "linear_trend": select_filter_table['linear_trend'],
                "linear_trend_noise": select_filter_table['linear_trend_noise'],
                "linear_trend_sigma": select_filter_table['linear_trend_sigma'],
                "magnitude_percentage_ratio_20_10": select_filter_table['magnitude_percentage_ratio_20_10'],
                "magnitude_percentage_ratio_40_5": select_filter_table['magnitude_percentage_ratio_40_5'],
                "maximum_slope": select_filter_table['maximum_slope'],
                "mean": select_filter_table['mean'],
                "mean_variance": select_filter_table['mean_variance'],
                "median": select_filter_table['median'],
                "median_absolute_deviation": select_filter_table['median_absolute_deviation'],
                "median_buffer_range_percentage_10": select_filter_table['median_buffer_range_percentage_10'],
                "percent_amplitude": select_filter_table['percent_amplitude'],
                "skew": select_filter_table['skew'],
                "standard_deviation": select_filter_table['standard_deviation'],
                "stetson_K": select_filter_table['stetson_K'],
                "weighted_mean": select_filter_table['weighted_mean']
            }

    return feature_table_all

def alert_lc(table, diaObjectId, alert_lc_type='prvDiaSources', flux_ref='scienceFlux', add_mags=True, band='r'):
    """For a given diaObjectId, retrieve light curve data.
    
    Parameters:
    -----------
    - table: pd.DataFrame, the input table containing alert data
    - diaObjectId: int, the associated ID of an diaObjectId
    - alert_lc_type: str, the type of alert light curve to retrieve (default: 'prvDiaSources', options: 'prvDiaSources', 'prvDiaForcedSources')
    - flux_ref: str, the flux reference to use for magnitude calculations (default: 'scienceFlux')
    - add_mags: bool, whether to add magnitude columns to the output (default: True)
    - band: str, the photometric band to use (default: 'r')

    Returns:
    -------
    pd.DataFrame: A DataFrame containing the light curve data for the specified diaObjectId.

    Notes:
    ------
    [1]: See https://lsst.fink-portal.org/schemas for more information.
    """
    data = table[table['diaObjectId'] == diaObjectId][alert_lc_type]

    lc = pd.json_normalize(list(data.iloc[0])) # convert list of dicts to dataframe
    
    if add_mags: 
        lc = lc.assign(mag=lc[f'{flux_ref}'].apply(lambda x: (x * u.nJy).to(u.ABmag).value),
        magErr=lc[f'{flux_ref}Err'].apply(lambda x: 1.086 * (x / lc[f'{flux_ref}'].iloc[0])))

    if band=='all':
        return lc
    else:
        return lc[lc['band'] == band]

def plot_alert_lc(alert_lc_table, band='r', yaxis='mag'):
    """Plot alert light curves for a given band and y-axis variable.
    """

    LSST_colors = ["#273F9B", "#2F8D2F", "#C9424B", "#FBA51B","#A50DB0", "#151616"]
    LSST_bands = list("ugrizy")

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 5))
    for b in band:
        flt = alert_lc_table['band'] == b
        if flt.sum() > 0:
            ax.errorbar(alert_lc_table.midpointMjdTai[flt], alert_lc_table[f'{yaxis}'][flt], yerr=alert_lc_table[f'{yaxis}Err'][flt], fmt='o', label=LSST_bands[LSST_bands.index(b)], color=LSST_colors[LSST_bands.index(b)])
    ax.set_xlabel("Time [MJD]")
    ax.set_ylabel(f"{yaxis}")
    if yaxis == 'mag':
        ax.invert_yaxis()

    ax.legend(ncols=6, frameon=True)
    plt.show()
