export type SunPosition = {
  timestamp: string
  latitude: number
  longitude: number
  direction: [number, number, number]
}
const radians = Math.PI / 180
const wrap = (angle: number) => ((((angle + 180) % 360) + 360) % 360) - 180

// USNO approximate solar coordinates and Greenwich sidereal time. UTC is an
// adequate approximation to UT1 here (visualization, not navigation).
// https://aa.usno.navy.mil/faq/sun_approx
// https://aa.usno.navy.mil/faq/GAST
export function solarPosition(time: Date): SunPosition {
  const milliseconds = time.getTime()
  if (!Number.isFinite(milliseconds)) throw new Error('Invalid solar timestamp.')
  const jd = milliseconds / 86400000 + 2440587.5
  const days = jd - 2451545
  const anomaly = wrap(357.529 + 0.98560028 * days) * radians
  const meanLongitude = wrap(280.459 + 0.98564736 * days)
  const longitude = (meanLongitude + 1.915 * Math.sin(anomaly) + 0.02 * Math.sin(2 * anomaly)) * radians
  const obliquity = (23.439 - 0.00000036 * days) * radians
  const rightAscension = Math.atan2(Math.cos(obliquity) * Math.sin(longitude), Math.cos(longitude)) / radians
  const latitude = Math.asin(Math.sin(obliquity) * Math.sin(longitude)) / radians
  const midnight = Math.floor(jd - 0.5) + 0.5
  const hours = (jd - midnight) * 24
  const centuries = days / 36525
  const sidereal =
    15 * (6.697375 + 0.065709824279 * (midnight - 2451545) + 1.0027379 * hours + 0.0000258 * centuries ** 2)
  const subsolarLongitude = wrap(rightAscension - sidereal)
  const lon = subsolarLongitude * radians,
    lat = latitude * radians
  return {
    timestamp: time.toISOString(),
    latitude,
    longitude: subsolarLongitude,
    direction: [Math.cos(lat) * Math.sin(lon), Math.sin(lat), Math.cos(lat) * Math.cos(lon)],
  }
}
