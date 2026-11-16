FROM earth-index-importer:latest
RUN pip install --no-cache-dir 'numcodecs==0.16.5'
# Freeze processors in the image so editing the repository during an overnight
# job cannot change its scientific computations or invalidate resume identity.
COPY scripts /workspace/scripts
COPY db/WORLD_IMPORT.md /workspace/db/WORLD_IMPORT.md
ENTRYPOINT ["python", "scripts/world_import.py"]
CMD []
