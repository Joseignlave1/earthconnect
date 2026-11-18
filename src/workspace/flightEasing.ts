type FlightEasing = (time: number) => number
type EasingPoint = readonly [time: number,progress: number]
type FlightPoints = readonly EasingPoint[]

export const flightDuration = 2_750
export const smootherStep: FlightEasing = time => time ** 3 * (10 - 15 * time + 6 * time * time)

// The opening preserves the selected timing from the earlier five-second flight.
const referenceDuration = 5_000
const referencePoints: FlightPoints = [[.1670524919172749,.235034],[.31,.741808],[.49,.998]]
const opening = warpClock(smootherStep,referencePoints)
const settleStart = flightDuration - 500
const [lastTime,lastProgress] = referencePoints[referencePoints.length - 1]
const lastAnchor = lastTime * referenceDuration
const settleProgress = opening(settleStart / referenceDuration)
const settleSpeed = (opening((settleStart + .01) / referenceDuration) -
  opening((settleStart - .01) / referenceDuration)) / .02
const tailDuration = flightDuration - lastAnchor
const tailSpeed = 2 * (1 - lastProgress) / tailDuration
const handoffDuration = lastAnchor - settleStart
const handoffMean = (lastProgress - settleProgress) / handoffDuration
const handoffExponent = (settleSpeed - tailSpeed) / (handoffMean - tailSpeed) - 1

export const baselineProgress: FlightEasing = time => {
  if (time <= 0 || time >= 1) return Math.max(0,Math.min(1,time))
  const elapsed = time * flightDuration
  if (elapsed <= settleStart) return opening(elapsed / referenceDuration)
  if (elapsed < lastAnchor) {
    // Integrate a decreasing speed that reaches the final user anchor
    // at the tail's entry speed, without stopping or speeding up again.
    const fraction = (elapsed - settleStart) / handoffDuration
    const excess = (settleSpeed - tailSpeed) * (1 - (1 - fraction) ** (handoffExponent + 1)) / (handoffExponent + 1)
    return settleProgress + handoffDuration * (tailSpeed * fraction + excess)
  }
  const remaining = (flightDuration - elapsed) / tailDuration
  return 1 - tailSpeed * tailDuration * remaining ** 3 * (1 - remaining / 2)
}

function warpClock(base: FlightEasing,points: FlightPoints): FlightEasing {
  // Interpolate a monotone clock through the reference points. Composing
  // it with smootherStep retains zero speed at both endpoints.
  const knots: EasingPoint[] = [[0,0],...points.map(([time,progress]): EasingPoint => [time,timeAtProgress(base,progress)]),[1,1]]
  const speed = knots.slice(1).map(([x,y],i) => (y - knots[i][1]) / (x - knots[i][0]))
  const tangent = knots.map((_k,i) => i === 0 ? speed[0] : i === knots.length - 1 ? speed[i - 1]
    : 2 / (1 / speed[i - 1] + 1 / speed[i]))
  return time => {
    if (time <= 0 || time >= 1) return Math.max(0,Math.min(1,time))
    const i = knots.findIndex(([x]) => x > time) - 1
    const [x,a] = knots[i], [next,b] = knots[i + 1]
    const width = next - x
    const t = (time - x) / width
    const easedTime = a + (b - a) * t * t * (3 - 2 * t) +
      width * t * (1 - t) * ((1 - t) * tangent[i] - t * tangent[i + 1])
    return base(easedTime)
  }
}

function timeAtProgress(easing: FlightEasing,progress: number) {
  if (progress <= 0 || progress >= 1) return Math.max(0,Math.min(1,progress))
  let low = 0,high = 1
  for (let i = 0; i < 44; i++) {
    const middle = (low + high) / 2
    if (easing(middle) < progress) low = middle
    else high = middle
  }
  return (low + high) / 2
}
