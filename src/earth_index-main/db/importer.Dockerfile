FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90
RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir \
    'rasterio==1.5.2' 'numpy==2.5.4' 'exactextract==0.3.0' \
    'shapely==2.2.0' 'pyproj==3.8.0' 'h3==4.5.0' 'pyhdf==0.11.7' 'h5py==3.16.0'
ENV PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
WORKDIR /workspace
ENTRYPOINT ["python", "scripts/thermal_import.py"]
CMD ["--help"]
