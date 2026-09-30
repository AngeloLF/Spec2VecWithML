from astroquery.simbad import Simbad
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
import astropy.units as u
import gaiaspec
from gaiaxpy import calibrate
from scipy.interpolate import interp1d
from tqdm import tqdm

import numpy as np
import pandas as pd
import traceback
import sys, os
import coloralf as c

Gaia.ROW_LIMIT = 10000
Simbad.add_votable_fields('G')



def query_HD(hd_name):

    result = Simbad.query_object(hd_name)

    ra0 = result["ra"][0]
    dec0 = result["dec"][0]
    mag0 = result["G"][0]
    gaia0 = get_gaia_id(hd_name)

    return ra0, dec0, mag0, gaia0




def add2ccd(xx, yy, star, tel, ra0, dec0, psf_func):

    x_c = (star.get("ra") - ra0) * 3600 / tel.get("CCD_PIXEL2ARCSEC") + tel.get("R0")[0]
    y_c = (star.get("dec") - dec0) * 3600 / tel.get("CCD_PIXEL2ARCSEC") + tel.get("R0")[1]
    
    flux = 10**(-star.get("mag_G") / 2.5) * 1e10

    return psf_func(xx, yy, amplitude=flux, x_c=x_c, y_c=y_c, gamma=3.0, alpha=2.0)


def get_gaia_id(hd_name):

    result = Simbad.query_objectids(hd_name)
    
    if result is None:
        print(f"Objet '{hd_name}' introuvable dans SIMBAD.")
        return ""

    for row in result:
        name = str(row["id"])
        if name.startswith("Gaia DR3"):
            return int(name.replace("Gaia DR3", ""))  # ex: "Gaia DR3 3510294882898890880"

    print(f"Aucun identifiant Gaia DR3 trouvé pour '{hd_name}'.")

    return ""




def get_gaia_neighbors(ra0, dec0, radius_deg):

    center = SkyCoord(ra=ra0, dec=dec0, unit=(u.deg, u.deg), frame="icrs")
    radius = u.Quantity(radius_deg*60, u.arcmin)
    job = Gaia.cone_search_async(center, radius=radius)
    table = job.get_results()

    neighbors = []
    for row in table:
        
        neighbors.append({
            "source_id": str(row["source_id"]),
            "ra":        round(float(row["ra"]), 8),
            "dec":       round(float(row["dec"]), 8),
            "mag_G":     round(float(row["phot_g_mean_mag"]), 4),
        })

    neighbors.sort(key=lambda x: x["ra"])

    return neighbors



def get_gaia_rect(ramin, ramax, decmin, decmax):

    query = f"""
        SELECT source_id, ra, dec, phot_g_mean_mag
        FROM gaiadr3.gaia_source
        WHERE ra  BETWEEN {ramin}  AND {ramax}
        AND   dec BETWEEN {decmin} AND {decmax}
    """

    job = Gaia.launch_job_async(query)
    table = job.get_results()

    neighbors = []
    for row in table:
        
        neighbors.append({
            "source_id": str(row["source_id"]),
            "ra":        round(float(row["ra"]), 8),
            "dec":       round(float(row["dec"]), 8),
            "mag_G":     round(float(row["phot_g_mean_mag"]), 4),
        })

    neighbors.sort(key=lambda x: x["ra"])

    return neighbors




def get_gaia_spec(source_ids, sampling_nm=np.arange(300, 1100), folder_gaia="./specSimulator/datafile/gaia"):

    already_gaia = os.listdir(folder_gaia)
    source_save = list()
    source_query = list()

    for sid in source_ids:

        if f"{sid}.npy" in already_gaia:
            source_save.append(sid)
        else:
            source_query.append(sid)
    
    try:

        if len(source_query) > 1:
            
            print(f"{c.m}Nb spectra to download : {len(source_query)}{c.d}")
            spectra, wavelengths = calibrate(source_query, save_file=False)

            # Create DataFrame
            spectra_df = pd.DataFrame(spectra)
            spectra_df.set_index('source_id', inplace=True)

            # save spectra_df
            pbar = tqdm(total=len(spectra_df), desc="query gaia")
            for sid, row in spectra_df.iterrows():
                func_flux = interp1d(wavelengths, row["flux"] * 1e3, kind='linear', bounds_error=False, fill_value=0.)(sampling_nm) # W/m2/nm -> 1e3 erg/s/cm2/nm
                func_flux_err = interp1d(wavelengths, row["flux_error"] * 1e3, kind='linear', bounds_error=False, fill_value=0.)(sampling_nm)
                npy = np.array([sampling_nm, func_flux, func_flux_err])
                np.save(f"{folder_gaia}/{sid}.npy", npy)
                pbar.update(1)
            pbar.close()

        else:

            print(f"{c.y}INFO [gaiaspec] : all gaia already download{c.d}")

    except ValueError as e:

        print(f"{c.lr}Error from gaiaXpy when all source_query did'n exist ...{c.d}")
        
    except Exception as e:
        print(f"Error fetching spectra: {e}")
        print(traceback.format_exc())
