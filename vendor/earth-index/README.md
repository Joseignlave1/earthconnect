# Earth Index client files

These files come from the Earth Index snapshot imported by Earth Connect in
[commit 503ffda](https://github.com/Joseignlave1/earthconnect/tree/503ffdad8648871e09f6ba87498d24e8731bfc1f/src/earth_index-main).

- `proto/` contains the API contract and its protobuf dependency. The site and
  score snapshot utility share the generated `earth_index_pb.ts` client types.
- The [4 km source manifest](https://github.com/Joseignlave1/earthconnect/blob/503ffdad8648871e09f6ba87498d24e8731bfc1f/src/earth_index-main/research/expansion-4km/manifest.json)
  records field units and provenance for the frozen 4 km dataset.
- `planetary-sources.json` records the layer sources used by the Moon and Mars
  sampler. See [Data sources and methodology](DATA_SOURCES.md) for dataset
  attribution and coverage.

Earth Index source code is distributed under its [Apache 2.0 license](LICENSE.txt).
The protobuf dependency retains its own license notice.
