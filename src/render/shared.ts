// Render hooks for other visual layers (atmosphere: clouds, ash/steam plumes, rain; life; overlays).
// One import point so they do not reach into individual render modules. All are read-only on sim
// state (V15).
//
// Clocks and light
//  - ambTime          uniform float, ambience seconds wrapped to AMB_PERIOD (V11, V17). Animate with
//                     it; scroll speeds as k / AMB_PERIOD keep the wrap seamless.
//  - skyU             shared sky uniforms, updated with the time of day: sunDir (vec3, towards the
//                     sun), sunColor (linear), sunIntensity, moonDir, night (0 day … 1 night),
//                     zenith / horizon / ground sky colours.
//  - skyColor(dir, discs?)  TSL sky radiance in a world direction (same sky as backdrop + water
//                     reflections); discs adds sun/moon/stars.
//  - viewDirWorld     TSL unit vector camera → fragment (world).
//
// Depth / scene access (transparent materials: soft particles, fog volumes)
//  - sceneViewZ(uv)   view-space z of the opaque scene at a screen uv (viewportDepthTexture, copied
//                     before transparents draw). Soft-particle fade:
//                       saturate(positionView.z.sub(sceneViewZ(screenUV)).mul(k))
//                     (positionView.z − sceneZ > 0 while the scene is behind the particle).
//  - viewportSharedTexture(screenUV) from 'three/tsl' gives the opaque scene colour (shared with
//                     the water refraction; one copy per frame).
//
// Placement and ordering
//  - Mapping: HALF (block half size), voxelToWorldY / tWorldY (voxel y → world, follows vertEx),
//    Y_RENDER_BOTTOM (lowest drawn voxel layer).
//  - Transparent order: water sheet renderOrder 2, glass water on the cut faces 3. Put volumes and
//    particles at ≥ 4 with depthWrite false. Emissive > ~1.1 (linear HDR) blooms.
//  - Objects added to stage.scene are drawn by the post pipeline's scene pass automatically.
export { ambTime, AMB_PERIOD, viewDirWorld, HALF, voxelToWorldY, tWorldY, Y_RENDER_BOTTOM, vertEx } from './space';
export { skyU, skyColor, GOLDEN_HOUR } from './sky';
export { sceneViewZ } from './water';
