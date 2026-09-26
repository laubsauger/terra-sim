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
- keys: `H` toggle UI, `T` tectonics layer, `Space` pause geo clock, `[`/`]` speed ÷2/×2, `I` inspect probe, `1`-`9` overlays, `0` overlay off, `M` mute
- ui: Tweakpane panes: Sim, Tectonics, Climate, Events, Overlays, God, Stats, Save
- time ctl: always-visible bar (hidden in ambient mode): pause/play geo, log speed slider (0.001 → max My/s), requested vs effective speed, `geoTime` readout
- overlays: 1 plates, 2 crust age, 3 surface temp, 4 mantle heat flow, 5 tectonic activity, 6 rivers & lakes, 7 precip, 8 biomes, 9 elevation; legend card bottom-left w/ hover readout; Tectonics layer (plate lines + velocity arrows, 'Plates' pill / key T) on top of normal render
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
V24: cont-cont collision → loser crust stacks onto winner (root down, isostatic rise) up to 64 layers; only excess delaminates. arc volcanism ! add FLAG_CONTINENTAL crust from reservoir (continents regrow). colliding continental plates lock (velocities converge); thick roots flow laterally into thinner continental neighbours.
V25: plate speed drivers ⊥ positive feedback on own speed; slab pull normalised by plate speed.
V26: water quasi-steady: overdamped flow + global open-ocean leveling (zero-sum, quantised); open-ocean level std ≤ 0.3 & neighbour roughness ≤ 0.05 voxel on the live sim.
V27: standing water deeper than ~2 voxels ⊥ erosion capacity; sediment settles (deltas, shelves). Σ sedSusp stays ≤ ~2 layers/col.
V28: thermal talus ≥ 2.5 layers/cell subaerial, ×2 submarine; stretched margins taper via lower-crust flow into oceanic neighbours.
V29: reservoir debt ⊥ grows unbounded: accretion takes only slab excess over ridge mass; collided crust stacking fades to 0 over 1 layer/col of debt (window snapshot, deterministic).
V30: ∀ column surfY ≥ deepest initial crust base & y=0 always PERIDOTITE (no holes through the world).
V31: every GPU test fails on WGSL / pipeline validation errors: a kernel that does not compile must never pass as a sim that merely does less.
V33: talus throughput ≥ 1 layer / direction / erosion step: walls at convergent fronts slump faster than the front rebuilds them; chamber roofs stay ≥ depthMin below the surface.
V34: every trench accretes slab excess (continental AND oceanic winners); arc + hotspot melt scale with reservoir fertility (clamp(res / 1 layer·col, 0.25, 4)) — mass-coupled rate, no target value.
V35: continents persist with live volcanism: over 1000 My continental mass stays ≥ ~75 % of initial, the reservoir ≈ 0-2 layers/col, eruptions never stall for > 100 My; plates stay 4-10 most of the time.
V36: no shader reads a TSL temp assigned only inside another branch (tests/gpu/shaders.spec.ts checks every WGSL module the app compiles).
V32: displayed plate motion moves every frame at ≈ plate speed; sim steps (every TEC_EVERY ticks, bursts at stats-window stalls) glide over the measured step interval.
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
T12|x|M1 inspect probe: raycast + small readback panel|I.probe
T13|x|M2 plate table + kinematics (slab pull, ridge push, drag)|V12
T14|x|M2 plate advect: sub-cell accum, column gather shift w/ wrap, time-sliced|V1,V19
T15|x|M2 convergent resolve: subduction, orogeny, trench/arc tagging|V3
T16|x|M2 divergent fill: new BASALT, ridge uplift, age reset|V3
T17|x|M2 isostasy column adjust|-
T18|x|M2 plate lifecycle: split, merge (suture), absorb tiny|V6
T19|x|M2 crust mass budget: subduction → `mantleReservoir` → ridges|volcanism|V3,V2
T20|x|M3 mantle field: diffuse/advect, plumes, slab sinks, side render glow|V19
T21|x|M3 crust temp field half-res, geotherm|-
T22|x|M3 melt gen (ridge, arc, hotspot), ascent, chambers|V3
T23|x|M3 eruption + 2D lava flow + cool/solidify into voxels|V3
T24|x|M4 shallow water pipe model, torus wrap, quasi-steady substeps|V1,V4
T25|x|M4 hydraulic erosion/deposit (`kGeo`) + thermal erosion|V3
T26|x|M4 sediment layering into voxels, diagenesis, metamorphism|V21
T27|.|M4 water budget: ocean+surface+vapor+ice, numeric renormalize|V4
T28|x|M5 surfTemp: latitude cos, lapse, ice-age offset|-
T29|x|M5 wind bands, vapor advect, orographic precip, rain shadow|V4
T30|x|M5 snow/ice accumulation, glaciers|V4
T31|x|M5 biome classify + veg; veg lowers erodibility|-
T32|x|M6 Wilson cycle controller (disperse→drift→assemble→super→rift)|V5,V8
T33|x|M6 seeded event scheduler: hotspot, meteor, ice age, flood basalt|V2,V16
T34|x|M6 stats reduction fixed-point + async readback + Stats pane|V2,V3,V4
T35|x|M6 NaN guard + autosave rollback|V7
T36|x|M6 soak harness (headless, max speed) asserting bounds|V8
T37|x|M7 water shading: absorption, refraction, caustics, foam, Gerstner, rivers|V17
T38|x|M7 lava/magma emissive + cooling crust|-
T39|x|M7 sky scattering + day/night + sun/moon on `ambTime`|V17
T40|x|M7 shadows + GTAO + SSGI \| fallback|V9
T41|~|M7 volumetric clouds from vapor/precip|V9,V15
T42|~|M7 weather FX: rain, lightning, ash plumes, fog|V15
T43|x|M7 post: TRAA, bloom, tilt-shift DOF, grade LUT, vignette|V9
T44|.M7 terrain phase 2: surface-nets mesher, chunked dirty remesh, indirect draw|V9,V19
T45|x|M7 tiny life: instanced flora per biome, boids, critters, fish|V15
T46|x|M7 procedural ambient audio from sim stats|V14,V15
T47|x|M8 debug overlays|I.overlays
T48|x|M8 god tools via event path|V16,I.god tools
T49|x|M8 ambient mode: hide UI, cinematic auto cam, POI follow (eruptions, impacts)|I.keys
T50|x|M8 save/load: `.terra` format, gzip, IDB autosave ring, file export/import|V13,V21,I.file,I.idb
T51|~|M8 perf pass: budget per pass, time-slice tuning, 24h leak run|V9,V10,V19

## §B BUGS
id|date|cause|fix
B1|2026-09-25|tectonics decide kernel bound 13 storage buffers > adapter limit 10 ∴ pipeline invalid|V23
B2|2026-09-25|cont-cont collision kept 1 layer of loser, rest → reservoir ∴ continents melted away ~150 My|V24
B3|2026-09-25|slab pull ∝ subducted cols ∝ own speed → runaway, ∀ plates @ max speed|V25
B5|2026-09-25|collision front saturates @ crust cap → ∀ further colliding crust deleted; plates ground continents @ full speed|V24
B6|2026-09-25|deep ocean kept full erosion capacity → sedSusp piled to 136 layers/col, reservoir overflowed int32|V27
B7|2026-09-25|plates ~2 cells/My on 256-cell world → continents collide constantly, area halves in 50 My|V25
B8|2026-09-25|no return path for eroded continental crust → continents thin & drown ~400 My|V24
B9|2026-09-25|talus 1.2 layer/cell (≈0.9° real) + underwater slumping → margins slump into sea forever|V28
B10|2026-09-25|ridges draw fixed mass per gap while collisions/accretion withhold loser mass → reservoir debt → runaway continents|V29
B11|2026-09-25|crustFlow treated crust-less neighbour (base=NY) as receiver → GNEISS written at ceiling → column sank to y=0 (holes)|V30
B12|2026-09-25|oceanic subsidence √age uncapped; stripped old continental columns (age 3000 My) sank through mantle|V30
B13|2026-09-25|tectonic water carry re-roughened ocean faster than local pipe model levels it → lumpy sea surface (level std ~2 voxels)|V26
B14|2026-09-25|overlapping columns pooled all candidates' water onto winner → continents overriding sea threw up water mounds|V26
B15|2026-09-26|margin crust flow let each new margin column become a sender → continents pancaked to ~24 layers, floating at sea level (world flattened just under water)|V28
B16|2026-09-26|collision lock averaged colliding plates' velocity vectors + merges averaged too → plates stalled (0.1-0.4 cells/My), headings swung, plate count collapsed|V25
B17|2026-09-26|orogeny raised surface by half the stacked layers in one run → mountains popped up instantly|V24
B18|2026-09-27|diverging continental plates opened one-cell ocean ridge slits inside continents → vertical flicker streaks on cut faces; rift fill (RIFT_THIN × neighbours, ≥ RIFT_MIN_THICK, ≤ RIFT_MAX_THICK, reservoir-gated) stretches crust instead|V29
B19|2026-09-27|TSL dropped a float cast inside uMax/select (f32 / u32) → tectonics voxel kernel invalid WGSL, never ran; plates, crust and land decayed (cont 0.39 → 0 in 50 My), all tests green|V31
B21|2026-09-27|talus capped at 63 fill units (0.25 layer) per direction per step + magma chambers growing to the surface (magma cannot slump) → 50-layer walls along every convergent front|V33
B22|2026-09-27|faster slumping fed trench slabs; ocean-ocean trenches accreted nothing and arc return counted nominal slabs → reservoir 1 → 7 layers/col, land 27 → 13 %|V34
B23|2026-09-27|collision fronts pinned at the 100-layer cap delaminated ~1/3 of colliding continental crust (jammed plates never slowed below MIN_SPEED, crust flow 1 layer/face) → continents drained into the mantle; reservoir at ~0 with melt floor starved volcanism|V35
B24|2026-09-27|oceanic trench winners: contMass − bestMass uint underflow stacked OROGENY_MAX per run at every ocean-ocean trench|V34
B25|2026-09-27|TSL assigns a node where first built: inside a branch, other reads see 0 → plume spawn dir/seed/wind 0 for most particle kinds (blobs, vertical fountains), flora base y = 0 with culling off|V36
B26|2026-09-27|sea cliffs stood forever: coastal talus measured against the seabed (B9 guard) never cut dry land facing water; now wave erosion cuts toward the water surface (never below sea)|V33
B27|2026-09-27|TRAA sub-pixel jitter never settled (shader-animated vertices lack motion vectors): ~3.8k pixels/frame jumped even paused, whole diorama shimmered|V36
B20|2026-09-27|plate display offset followed sim steps instantly; at normal speed a 0.1-0.3 cell step every few frames read as jerking plates|V32
B4|2026-09-25|pipe model friction 0.02 → deep ocean rang w/ persistent waves, level rough ~5 voxels|V26
