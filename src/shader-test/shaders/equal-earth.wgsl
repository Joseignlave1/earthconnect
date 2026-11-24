// Equal Earth (Šavrič et al., 2018): strictly equal-area pseudocylindrical projection (~2.05:1).
fn equalEarth(lon: f32, lat: f32) -> vec2f {
  let theta = asin(clamp(.8660254038 * sin(lat), -1., 1.));
  let t2 = theta * theta;
  let t6 = t2 * t2 * t2;
  let d = 1.340264 - .243318 * t2 + t6 * (.006251 + .034164 * t2);
  return vec2f(
    1.1547005384 * lon * cos(theta) / d,
    theta * (1.340264 - .081106 * t2 + t6 * (.000893 + .003796 * t2)),
  );
}
