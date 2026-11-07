export type Mode = 'off' | 'in' | 'out'
export type Wall = 'front' | 'rear' | 'top' | 'bottom'

export interface Mount { slot: string; wall: Wall; a: number; b: number; sizes: number[] }
export interface CasePart {
  id: string; name: string; brand: string; aliases: string[]; dims_mm: { D: number; W: number; H: number }
  max_gpu_len_mm: number; max_cooler_h_mm: number; mounts: Mount[]; preinstalled: { slot: string; fan: string; mode: Mode }[]
  sources: string[]; amazon_query?: string
}
export interface GpuPart { id: string; name: string; brand: string; aliases: string[]; length_mm: number; tdp_w: number; cooler: string; sources: string[]; amazon_query?: string }
export interface CoolerPart { id: string; name: string; brand: string; aliases: string[]; type: string; height_mm: number; fans?: { fan: string }[]; sources: string[]; amazon_query?: string }
export interface FanPart { id: string; name: string; brand: string; size_mm: number; cfm: number; static_mmh2o: number; rpm_max: number }
export interface Lib { cases: CasePart[]; gpus: GpuPart[]; coolers: CoolerPart[]; fans: FanPart[] }

export interface FanSlot { slot: string; fan: string; mode: Mode; pct: number }
export interface Build { case: string; gpu: string; cooler: string; fans: FanSlot[]; load: { cpu_w: number; gpu_w: number; ambient_c: number } }

export interface HW {
  cpu: { name: string }; board: string; ram_gb: number
  gpus: { name: string; temp_c: number | null; power_w: number | null; fan_pct: number | null }[]
  cpu_temp_c: number | null
}

export interface PartRes { intake_rise_c: number; temp_c: number; watts: number; calibrated?: boolean }
export interface SimFan { slot: string; mode: 'in' | 'out'; pct: number; wall: Wall; a: number; b: number; size: number; name: string; q_m3s: number; op_frac?: number }
export interface SimRes {
  label: string; build?: Build; shape: [number, number, number]; off: number; dx_mm: number
  case: { D: number; W: number; H: number }; case_name: string
  mounts: Mount[]; panels: Record<string, string>; fans: SimFan[]
  boxes: { name: string; kind: string; b: number[] }[]
  u: string; T: string; parts: { cpu: PartRes; gpu: PartRes }
  pressure: string; flow_in_m3s: number; flow_out_m3s: number; case_pa?: number
  dead_air_fraction: number; air_max_rise_c: number; seconds: number; notes: string[]
  dust: null | { dust_entering_rel: number; deposited_frac: number; d_um: number; by_part: { name: string; share: number }[]; points: number[][] }
}

export interface RankRow {
  name: string; build: Build; current?: boolean; gpu_c: number; cpu_c: number; score: number; pressure: string
  dust_in: number; combined?: number; cfm_in: number; cfm_out: number
}
export interface Opt {
  ranking: RankRow[]; best: string; coolest?: string; cleanest?: string | null; cpu?: string | null; worst?: string
  objective: string; dust_weight?: number; space?: string; minutes?: number
}

export type ViewKey = 'current' | 'best' | 'cpu' | 'worst' | 'custom'

export const WHERE: Record<string, string> = {
  front_low: 'front-low', front_high: 'front-high', bottom_front: 'bottom-front', bottom_rear: 'bottom-rear',
  top_1: 'top-front', top_2: 'top-mid', top_3: 'top-rear', rear: 'rear',
}
