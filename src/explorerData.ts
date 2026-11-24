export type BodyId = 'moon' | 'mars'

export type Criterion = {
  key: string
  label: string
  unit: string
  group: string
  enabled: boolean
  target: number | null
  tolerance: number
  weight: number
  min: number | null
  max: number | null
  description?: string
  categorical?: boolean
  qualityKey?: string | null
  origin?: string
  kind?: 'measured' | 'approximate' | 'preference' | 'manual'
  selectionTouched?: boolean
  selectedByUser?: boolean
  toleranceTouched?: boolean
  weightTouched?: boolean
  targetTouched?: boolean
}
