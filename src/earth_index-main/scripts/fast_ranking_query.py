"""SQL source optimization for the fixed, aligned 100x100 raster pilot.

Keep the original scorer intact. Expand just the required raster bands in
row-major order, preserving NULLs and pixel coordinates, before applying exactly
the same gate, transforms, weights and window aggregation.
"""
from psycopg import sql

from evaluate_ranking_pilot import query as original_query

ORIGINAL_SOURCE='SELECT * FROM research.priority_pilot_cells WHERE bundle_sha256=%s'
GATE_FIELDS={'bare_fraction','landcover_valid_fraction','slope_mean_deg','terrain_valid_fraction'}


def required_fields(spec,weights):
    fields=set(GATE_FIELDS)
    for component in weights:
        definition=spec['components'][component];fields.add(definition['field'])
        if 'coverage_field' in definition:fields.add(definition['coverage_field'])
    return sorted(fields)


def cell_source(spec,weights):
    fields=required_fields(spec,weights);ids=[];columns=[];joins=[];expansions=[sql.SQL('generate_series(1,10000)')]
    aliases=[sql.Identifier('pixel')];values=[]
    for i,field in enumerate(fields):
        layer=sql.Identifier('layer_'+str(i));tile=sql.Identifier('t'+str(i));codes=sql.Identifier('codes_'+str(i))
        scale=sql.Identifier('scale_'+str(i));offset=sql.Identifier('offset_'+str(i));raw=sql.Identifier('raw_'+str(i))
        ids.append(sql.SQL('max(layer_id) FILTER (WHERE feature_key={}) AS {}').format(sql.Literal(field),layer))
        joins.append(sql.SQL('LEFT JOIN research.raster_tiles {} ON {}.layer_id=p.{}').format(tile,tile,layer))
        columns.extend([sql.SQL('ST_DumpValues({}.rast,1,true) AS {}').format(tile,codes),
                        sql.SQL('{}.scale AS {}').format(tile,scale),sql.SQL('{}.offset_value AS {}').format(tile,offset)])
        expansions.append(sql.SQL('unnest(d.{})').format(codes));aliases.append(raw)
        values.append(sql.SQL('u.{}*d.{}+d.{} AS {}').format(raw,scale,offset,sql.Identifier(field)))
    # ROWS FROM zips arrays rather than cross-joining them. The ordinal generator
    # keeps every pixel even if one or all requested bands are absent. Band
    # NODATA becomes NULL before scale/offset decoding, preserving physical zero.
    return sql.SQL('''WITH packed AS MATERIALIZED (
        SELECT site_key,{ids} FROM research.raster_layers
        WHERE bundle_sha256=%s GROUP BY site_key
    ), decoded AS MATERIALIZED (
        SELECT p.site_key,{columns} FROM packed p {joins}
    )
    SELECT d.site_key,mod(u.pixel-1,100)+1 AS x,((u.pixel-1)/100)+1 AS y,{values}
    FROM decoded d CROSS JOIN LATERAL ROWS FROM ({expansions}) AS u({aliases})''').format(
        ids=sql.SQL(',').join(ids),columns=sql.SQL(',').join(columns),joins=sql.SQL(' ').join(joins),
        values=sql.SQL(',').join(values),expansions=sql.SQL(',').join(expansions),aliases=sql.SQL(',').join(aliases))


def query(spec,weights,connection):
    text=original_query(spec,weights).as_string(connection)
    if text.count(ORIGINAL_SOURCE)!=1:raise ValueError('Original scoring template changed; review the source substitution')
    return sql.SQL(text.replace(ORIGINAL_SOURCE,cell_source(spec,weights).as_string(connection)))
