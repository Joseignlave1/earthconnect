import { fieldsFor } from '../src/query.ts';
import { MeasurementStatus } from '../src/gen/earth_index_pb.ts';

export const columns = [
  'slope_mean_deg',
  'relief_5km_m',
  'relief_21km_m',
  'tpi_5km_m',
  'tpi_21km_m',
  'bare_fraction',
  'elevation_median_m',
  'coldest_month_air_c',
  'warmest_month_air_c',
  'terrain_valid_fraction',
  'landcover_valid_fraction',
];
export const datasource = {
  relations: [
    {
      name: 'cells',
      columns: columns.map((name) => ({
        name,
        unit: name.endsWith('_air_c')
          ? 'degC'
          : name.includes('slope')
            ? 'degree'
            : name.endsWith('_fraction')
              ? '1'
              : 'm',
      })),
    },
  ],
};
export const fields = fieldsFor(datasource);
export const newCriteria = () =>
  fields.map((f) => ({
    key: f.key,
    target: null,
    enabled: false,
    tolerance: f.tolerance,
    weight: 1,
    min: null,
    max: null,
  }));
export const measurement = (id, decoded, unit = 'm', extra = {}) => ({
  variableId: `mars.${id}`,
  status: MeasurementStatus.AVAILABLE,
  unit,
  decoded,
  qualityFlags: [],
  ...extra,
});
export const sample = {
  measurements: [
    measurement('slope_1km', 1, 'degree'),
    measurement('relief_5km', 100),
    measurement('relief_21km', 500),
    measurement('tpi_5km', 0),
    measurement('tpi_21km', -50),
    measurement('elevation', 5000),
    measurement('modeled_surface_temperature_min', -100, 'degC'),
  ],
};
export const target = { body: 'mars' };
