import { createClient } from '@connectrpc/connect';
import { createGrpcWebTransport } from '@connectrpc/connect-web';
import { EarthIndex } from './gen/earth_index_pb.ts';

export const api = createClient(
  EarthIndex,
  createGrpcWebTransport({
    baseUrl: globalThis.location?.origin ?? 'http://localhost:8080',
  }),
);
