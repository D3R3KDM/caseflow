import type { Bank, BankLevel, Curve, Curves, Pt } from './scenario'

export type Mode = 'off' | 'in' | 'out'
export type Wall = 'front' | 'rear' | 'top' | 'bottom'
export type Control = 'curve' | 'fixed'

export interface Mount { slot: string; wall: Wall; a: number; b: number; sizes: number[] }
export interface CasePart {
  id: string; name: string; brand: string; aliases: string[]; dims_mm: { D: number; W: number; H: number }
  max_gpu_len_mm: number; max_cooler_h_mm: number; mounts: Mount[]; preinstalled: { slot: string; fan: string; mode: Mode }[]
  sources: string[]; amazon_query?: string
}
export interface GpuPart {
  id: string; name: string; brand: string; aliases: string[]; length_mm: number; tdp_w: number; idle_w: number; t_max_c: number
  cooler: string; cooling: { fans: number; fan_rpm_max: number; fan_curve: Pt[] }; sources: string[]; amazon_query?: string
}
export interface CoolerPart { id: string; name: string; brand: string; aliases: string[]; type: string; height_mm: number; fans?: { fan: string; count?: number }[]; sources: string[]; amazon_query?: string }
export interface FanPart { id: string; name: string; brand: string; size_mm: number; cfm: number; static_mmh2o: number; rpm_max: number; s_min?: number; noise_dba?: number }
export interface CpuPart { id: string; name: string; brand: string; cores: number; tdp_w: number; max_w: number; idle_w: number; t_max_c: number; note?: string }
export interface Preset { id: string; name: string; cpu: number; gpu: number; note?: string }
export interface Lib {
  cases: CasePart[]; gpus: GpuPart[]; coolers: CoolerPart[]; fans: FanPart[]; cpus: CpuPart[]
  defaults: { ambient_c: number; curves: { case: Curve; cpu: Curve }; scenarios: Preset[] }
}

export interface FanSlot { slot: string; fan: string; mode: Mode; control?: Control; pct: number }
export interface Build {
  case: string; gpu: string; cooler: string; cpu?: string; fans: FanSlot[]; curves?: Partial<Curves>
  load?: { cpu_w: number; gpu_w: number; ambient_c: number }
}

/** A room temperature and a load, given as fractions between each part's idle power (0) and its limit (1). */
export interface Scenario { ambient_c: number; cpu: number; gpu: number; preset: string | null }

export interface Box { name: string; kind: string; b: number[] }
export interface GeoFan { slot: string; fan: string; mode: 'in' | 'out'; control?: Control; pct?: number; wall: Wall; a: number; b: number; size: number; name: string; filtered?: boolean }
export interface Geometry {
  shape: [number, number, number]; off: number; dx_mm: number; case: { D: number; W: number; H: number }; case_name: string
  mounts: Mount[]; panels: Record<string, string>; boxes: Box[]; checks: Record<string, unknown>; fans: GeoFan[]
}
/** An 8-bit field: value = (byte - 128) / 127 * scale (signed) or byte / 255 * scale. */
export interface Q8 { b64: string; scale: number; signed: boolean }
export interface BankPkg extends Bank {
  version: number; key: string; levels: BankLevel[]; curve_fans: string[]; geometry: Geometry; seconds?: number
  fields?: { u: Q8; kc: Q8; kg: Q8; dust_points: number[][] }[]
}

export interface Layout { name: string; build: Build; current: boolean; bank: Bank & { key: string; curve_fans: string[] } }
export interface Opt {
  version: number; layouts: Layout[]; dust_weight: number; tie_margin: number; space: string; objective: string; minutes: number
}
/** One layout's result in the current scenario (the search screen ranks these). */
export interface RankRow {
  name: string; build: Build; current: boolean; cpu_c: number; gpu_c: number; throttled: boolean; noise_dba: number
  case_duty: number; cfm_in: number; dust_rel: number; score: number; tied: boolean
}

export type ViewKey = 'current' | 'best' | 'cpu' | 'worst' | 'custom'

export const WHERE: Record<string, string> = {
  front_low: 'front-low', front_high: 'front-high', bottom_front: 'bottom-front', bottom_rear: 'bottom-rear',
  top_1: 'top-front', top_2: 'top-mid', top_3: 'top-rear', rear: 'rear',
}
