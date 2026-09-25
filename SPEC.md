# SPEC

## §G GOAL
whimsical AAA-look WebGPU vivarium: torus-wrapped planet slice block, voxel crust shaped by tectonics, magma, water, climate; inspectable, tweakable, runs unattended forever & stays interesting.

## §C CONSTRAINTS
- TS strict + Vite. three.js latest: `three/webgpu` WebGPURenderer, `three/tsl` node materials + compute. raw WGSL only where TSL lacks feature; comment why.
- target desktop Chrome|Edge, dGPU, 60fps @1440p. ref GPU = RTX 3060-class.
- WebGPU required. ⊥ WebGL fallback.
- deps minimal: `three`, `tweakpane` (+ essentials plugin). ⊥ physics engine. ⊥ UI framework.
- physics = plausible approximation, not real units. vertical exaggeration config const.
- longevity only via: conserved budgets, Wilson cycle controller, seeded random events. ⊥ homeostat (⊥ auto-tweak params toward "interesting").
- dual clock: `geoTime` (My, sim) & `ambTime` (s, waves/day-night/clouds/life/audio). independent.
- deterministic from seed. PRNG = PCG32, state serializable.
- art: stylized PBR diorama. tilt-shift, soft GI, saturated, glowing magma, caustic water. hi-fi but cartoony.
- tests: vitest (CPU logic + CPU reference kernels on small grids). Playwright + Chromium WebGPU (GPU kernels vs CPU ref, hash determinism, soak).
- git: ⊥ destructive cmds (checkout, reset). commits grouped per task, ⊥ micro-commits.

### data layout
- axes: X,Z horizontal (wrap, torus), Y up. voxel grid 256×256×128 (X×Z×Y).
- voxel u32 pack: `mat:8 | fill:8 | age:8 | flags:8`. `fill` = solid fraction 0-255 (smooth surface, sub-voxel erosion). `age` = log-scale code.
- voxel storage: 2× storage buffer ping-pong (~64MB). render reads storage buffers directly (both, parity uniform); ⊥ 3D texture copy.
- `mat` ids: AIR, BASALT, GABBRO, GRANITE, ANDESITE, SEDIMENT(loose), SANDSTONE, SHALE, LIMESTONE, SCHIST, GNEISS, MAGMA, PERIDOTITE(mantle). list append-only (save compat).
- column fields 2D 256² f32 (storage buffers): `plateId`(u8 packed), `surfY`(derived), `crustThick`, `crustAge`, `stress`, `water`, `flux`(4 pipes), `waterVel`(vec2), `sedSusp`, `lava`, `lavaTemp`, `surfTemp`, `vapor`, `precip`, `ice`, `veg`, `biome`(u8), `heatFlow`.
- crust temp 3D 128×128×64 f16. mantle 3D 64×64×32 (temp + velocity), lies below voxel grid.
- plate table SSBO ≤16 plates: `vel` vec2, `accum` vec2, `area`, `contFrac`, `age`, `flags`. CPU mirror authoritative for kinematics.
- budgets: `M_total` (crust mass: voxels + mantleReservoir + magma + lava + sedSusp), `W_total` (ocean+surface water + vapor + ice), heat input rate const. reductions → fixed-point int atomics.
- ∀ GPU buffers alloc once @ init, fixed size.

### sim pass order (per geo tick, fixed `dtGeo`)
1. controller (CPU): Wilson phase, event queue, plate velocity targets.
2. mantle: advect/diffuse temp, plumes, slab cold sinks. every k ticks, time-sliced.
3. plates: accumulate sub-cell offset; on cell cross → column gather shift w/ wrap; resolve convergent (subduction: denser/older oceanic sinks; cont-cont: thicken/orogeny); fill divergent gaps w/ new BASALT, age reset.
4. isostasy: column vertical shift toward equilibrium from thickness & density.
5. magma: melt where temp > solidus (ridge, arc offset from trench, hotspot); ascent; chamber fill; pressure > threshold → eruption event.
6. lava: 2D viscous flow, cool, solidify → voxels (cones build).
7. climate: surfTemp (latitude `cos(2πz/Z)` + lapse + ice-age offset), zonal wind bands, evaporation, vapor advect, orographic precip, snow/ice.
8. hydrology: shallow water pipe model (quasi-steady substeps) → hydraulic erosion/deposit scaled by `kGeo`; thermal erosion (talus angle); veg lowers erodibility.
9. diagenesis/metamorphism (slow): SEDIMENT → SANDSTONE|SHALE|LIMESTONE by env; depth+temp → SCHIST|GNEISS.
10. derived: `surfY`, normals, biome (Whittaker temp×precip), veg, dirty chunks.
11. stats: budget + world stats reduction every N ticks, async readback.

### render pipeline
- terrain: phase 1 heightfield mesh displaced by `surfY`, seamless wrap; phase 2 GPU surface-nets mesher on `fill`, chunks 32³, dirty remesh, indirect draw (caves, overhangs, lava tubes).
- materials: triplanar stylized PBR, splat by `mat` + biome + snow + wetness.
- side cuts: 4 faces + bottom read voxel storage buffer & mantle field: strata colors, magma glow, convection flow, crust temp tint.
- water: depth absorption, screen-space refraction, caustics on seabed (`ambTime`), shore foam, small Gerstner waves, rivers from `flux`.
- lava: emissive + cooling crust noise → bloom.
- sky: atmospheric scattering LUT, sun/moon, day/night on `ambTime`.
- lighting: directional + shadow map, GTAO, SSGI | hemisphere-AO fallback.
- clouds: raymarched volume in box above slice, half-res + temporal reprojection, density from `vapor`/precip.
- weather FX: rain particles, lightning, ash plumes, fog.
- post: TRAA, bloom, tilt-shift DOF, grade LUT, vignette.
- life: GPU-scattered instanced flora per biome, boids birds, tiny critters, fish. read-only on sim.

## §I INTERFACES
- cmd: `npm run dev` | `build` | `test` (vitest) | `test:gpu` (Playwright) | `soak` (headless max-speed run, asserts §V.8)
- url: `?seed=<u32>&speed=<x>&ambient=1&load=<slot>&quality=<low|high>`
- keys: `H` toggle UI, `Space` pause geo clock, `[`/`]` speed ÷2/×2, `I` inspect probe, `1`-`9` overlays, `0` overlay off, `M` mute
- ui: Tweakpane panes: Sim, Tectonics, Climate, Events, Overlays, God, Stats, Save
- time ctl: always-visible bar (hidden in ambient mode): pause/play geo, log speed slider (0.001 → max My/s), requested vs effective speed, `geoTime` readout
- overlays: plateId, crustAge, crustTemp, mantle, stress, flux, moisture, biome, precip
- god tools: uplift|subsidence brush, spawn volcano, meteor impact, split plate, rain storm
- probe: click → mat, fill, age, temp, plateId, elevation, water, biome, local history sparkline
- file: `*.terra` = header {magic `TERA`, version u16, seed u32, geoTime f64, rngState, params JSON, controller JSON} + gzip(buffers)
- idb: db `terra-sim`, store `saves`, autosave ring 3 slots, interval param (default 5 min)
- debug: `window.terra` = { stats(), hash(), step(n), save(), load(blob) } for tests

## §V INVARIANTS
V1: ∀ sim neighbor read on X,Z → index mod N (torus). ⊥ edge clamp in sim. seam ⊥ visible in render.
V2: same seed & params & device → identical `hash()` after N ticks. ⊥ float atomics; reductions fixed-point int.
V3: crust mass `M_total` conserved: rel drift ≤ 1e-4 per 1000 ticks.
V4: water `W_total` conserved: rel drift ≤ 1e-4 per 1000 ticks. renormalize only numeric error ≤ ε; ⊥ used as nudge.
V5: ⊥ runtime param auto-adjust. only Wilson controller & event scheduler & user change params/state.
V6: plate count ∈ [3, 12] always. split|merge|absorb keep bound.
V7: ∀ field finite. NaN|Inf detected ≤ 1s → rollback to last autosave + log event.
V8: soak 5000 My max speed → land frac ∈ [0.15, 0.6], ocean frac ≥ 0.3, relief ≥ `reliefMin`, ≥1 eruption per 50 My, ≥1 full Wilson cycle, V3 & V4 & V6 & V7 hold.
V9: ref GPU @1440p default quality → frame ≤ 16.6ms, sim GPU ≤ 5ms/frame.
V10: GPU memory fixed after init; ⊥ per-frame buffer|texture alloc. JS heap growth ≤ 5% over 24h run.
V11: `geoTime` f64 on CPU only. GPU gets `dtGeo` & `ambTime mod period` ∴ ⊥ precision decay over days.
V12: sim result ⊥ depends on frame rate. fixed `dtGeo`; fps changes ticks per frame only.
V13: save → load → identical `hash()`. version mismatch → reject w/ message, ⊥ partial load.
V14: audio starts only after user gesture. mute persists.
V15: life, audio, weather FX, post read sim only; ⊥ write sim state.
V16: god tools & random events enqueue through same event path; ∀ mass|water moved via budgets (uplift draws `mantleReservoir`) ∴ V3 & V4 hold.
V17: pause geo clock ⊥ pauses `ambTime` (waves, clouds, day/night, life keep moving).
V18: no WebGPU → error screen w/ reason. ⊥ blank page.
V19: heavy passes (plate shift, remesh, mantle) time-sliced; sim ⊥ adds frame spike > 33ms.
V20: params single schema {default, range, unit, persist}. UI, url, save, soak read same schema.
V21: `mat` id list append-only; saves from older version keep valid materials.
V22: speed change ⊥ changes `dtGeo`; only ticks per frame. effective speed = min(requested, sim budget V9); UI shows both.
V23: ∀ kernel & material ≤ 8 storage buffers bound (WebGPU default `maxStorageBuffersPerShaderStage`). pack fields (uvec2/vec4) & small tables → `uniformArray`.

## §T TASKS
id|status|task|cites
T1|x|M0 scaffold Vite+TS strict+three+tweakpane, vitest, Playwright WebGPU harness|I.cmd
T2|x|M0 renderer bootstrap: WebGPURenderer, orbit cam, WebGPU detect + error screen|V18
T3|x|M0 PCG32 PRNG w/ serializable state|V2
T4|x|M0 dual clock: `geoTime` f64, `ambTime` wrapped, pause geo only, speed → ticks/frame w/ budget cap|V11,V12,V17,V22
T5|x|M0 GPU buffer registry: fixed alloc, named fields, ping-pong, timestamp-query perf HUD|V10,V9
T6|x|M0 param schema + Tweakpane shell + url params + time ctl bar|V20,V22,I.url,I.ui,I.time ctl
T7|x|M1 voxel pack/unpack TSL helpers + CPU mirror, wrap index helpers|V1,V21
T8|x|M1 worldgen from seed: torus voronoi plates, cont|ocean crust stacks, strata|V2,V6
T9|x|M1 derived pass: `surfY` from `fill`, normals|V1
T10|x|M1 terrain render phase 1: heightfield mesh, triplanar splat, seamless wrap|V1
T11|x|M1 side cut render: faces + bottom read voxel buffer, strata colors|V23
T12|.|M1 inspect probe: raycast + small readback panel|I.probe
T13|x|M2 plate table + kinematics (slab pull, ridge push, drag)|V12
T14|x|M2 plate advect: sub-cell accum, column gather shift w/ wrap, time-sliced|V1,V19
T15|x|M2 convergent resolve: subduction, orogeny, trench/arc tagging|V3
T16|x|M2 divergent fill: new BASALT, ridge uplift, age reset|V3
T17|x|M2 isostasy column adjust|-
T18|.|M2 plate lifecycle: split, merge (suture), absorb tiny|V6
T19|.|M2 crust mass budget: subduction → `mantleReservoir` → ridges|volcanism|V3,V2
T20|.|M3 mantle field: diffuse/advect, plumes, slab sinks, side render glow|V19
T21|.|M3 crust temp field half-res, geotherm|-
T22|.|M3 melt gen (ridge, arc, hotspot), ascent, chambers|V3
T23|.|M3 eruption + 2D lava flow + cool/solidify into voxels|V3
T24|x|M4 shallow water pipe model, torus wrap, quasi-steady substeps|V1,V4
T25|x|M4 hydraulic erosion/deposit (`kGeo`) + thermal erosion|V3
T26|.|M4 sediment layering into voxels, diagenesis, metamorphism|V21
T27|.|M4 water budget: ocean+surface+vapor+ice, numeric renormalize|V4
T28|.|M5 surfTemp: latitude cos, lapse, ice-age offset|-
T29|.|M5 wind bands, vapor advect, orographic precip, rain shadow|V4
T30|.|M5 snow/ice accumulation, glaciers|V4
T31|.|M5 biome classify + veg; veg lowers erodibility|-
T32|.|M6 Wilson cycle controller (disperse→drift→assemble→super→rift)|V5,V8
T33|.|M6 seeded event scheduler: hotspot, meteor, ice age, flood basalt|V2,V16
T34|.|M6 stats reduction fixed-point + async readback + Stats pane|V2,V3,V4
T35|.|M6 NaN guard + autosave rollback|V7
T36|.|M6 soak harness (headless, max speed) asserting bounds|V8
T37|.|M7 water shading: absorption, refraction, caustics, foam, Gerstner, rivers|V17
T38|.|M7 lava/magma emissive + cooling crust|-
T39|.|M7 sky scattering + day/night + sun/moon on `ambTime`|V17
T40|.|M7 shadows + GTAO + SSGI \| fallback|V9
T41|.|M7 volumetric clouds from vapor/precip|V9,V15
T42|.|M7 weather FX: rain, lightning, ash plumes, fog|V15
T43|.|M7 post: TRAA, bloom, tilt-shift DOF, grade LUT, vignette|V9
T44|.|M7 terrain phase 2: surface-nets mesher, chunked dirty remesh, indirect draw|V9,V19
T45|.|M7 tiny life: instanced flora per biome, boids, critters, fish|V15
T46|.|M7 procedural ambient audio from sim stats|V14,V15
T47|.|M8 debug overlays|I.overlays
T48|.|M8 god tools via event path|V16,I.god tools
T49|.|M8 ambient mode: hide UI, cinematic auto cam, POI follow (eruptions, impacts)|I.keys
T50|.|M8 save/load: `.terra` format, gzip, IDB autosave ring, file export/import|V13,V21,I.file,I.idb
T51|.|M8 perf pass: budget per pass, time-slice tuning, 24h leak run|V9,V10,V19

## §B BUGS
id|date|cause|fix
B1|2026-09-25|tectonics decide kernel bound 13 storage buffers > adapter limit 10 ∴ pipeline invalid|V23
