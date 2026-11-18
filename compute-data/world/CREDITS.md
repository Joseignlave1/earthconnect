# Frozen Earth Index 4 km world

The score dataset uses the Earth Index datasource `earth-ed5bc9904a5f`, revision
`ed5bc9904a5f7211e52e3ac5b21cd145fe2f69f6f38ddc0df98da08a69e90f0e`. The original
25 quantized fields, 460 grid sites and all padded addresses are preserved.
The files are a repackaging of the Earth Index research export into GPU-readable
blocks with Zstandard delivery frames.

- **Terrain.** EarthEnv global topography derived from GMTED2010, Amatulli et al. (2018). [Dataset](https://doi.org/10.1594/PANGAEA.867113), [methods](https://doi.org/10.1038/sdata.2018.40).
- **Land cover.** NASA MODIS MCD12Q1 V061, 2024, Friedl and Sulla-Menashe (2022). [Product](https://doi.org/10.5067/MODIS/MCD12Q1.061).
- **Surface temperature.** NASA Terra MODIS MOD11A2 V061, 2023, Wan, Hook and Hulley (2021), accessed through Microsoft Planetary Computer. [Product](https://doi.org/10.5067/MODIS/MOD11A2.061), [access catalog](https://planetarycomputer.microsoft.com/dataset/modis-11A2-061).
- **Climate.** NASA POWER Project at NASA Langley Research Center, POWER v10.0.0 MERRA-2 2001–2020 climatology. [Bulk data](https://power.larc.nasa.gov/docs/services/aws/).
- **Permafrost.** ESA Climate Change Initiative, Permafrost CCI and data providers, Westermann et al. (2025), Northern Hemisphere permafrost extent v5.0, 2023. [Dataset](https://doi.org/10.5285/d235665772ec4b558e9a89ac85595e71).
- **Lithology.** GLiM, Hartmann and Moosdorf (2012), gridded v1.0 0.5° map. [Dataset](https://doi.org/10.1594/PANGAEA.788537), [methods](https://doi.org/10.1029/2012GC004370).
