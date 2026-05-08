// ── webgl_overlay.js ──────────────────────────────────────────────────────
// WebGL2 instanced renderer for detection overlays (points + bounding boxes).
//
// Uses a dedicated <canvas id="webgl-canvas"> stacked on top of mask-canvas,
// so it never conflicts with the Canvas2D context already held by maskCtx.
//
// Render model:
//   requestAnimationFrame loop → reads _packedDetections each frame →
//   uploads to VBO only when frameId or renderPoints flag changes →
//   single gl.drawArraysInstanced() call.
//
// Anti-flicker: because rendering is rAF-driven and never clears between
// SSE events, the overlay is always "present" from the latest buffer.
// No visible blank gap between clear and redraw.
//
// Programs:
//   progPointsCenter  – rowWidth=3  [cx,cy,conf] stride=12  → circles
//   progPointsBbox    – rowWidth=5, renderPoints=true        → circles (center computed in shader)
//   progBbox          – rowWidth=5, renderPoints=false       → outlined rectangles
//
// Exposes: window._webglOverlayActive (bool)
// Depends on: state.js  (renderPoints, showMask, previewOnly,
//                        _packedDetections, _packedDetectionsRowWidth,
//                        _packedDetectionsFrameId)
//             overlays.js (_computeLayout)
// ──────────────────────────────────────────────────────────────────────────

(function () {
  'use strict';

  const webglCanvas = document.getElementById('webgl-canvas');
  if (!webglCanvas) {
    window._webglOverlayActive = false;
    window.__overlayRenderer = 'canvas2d';
    return;
  }

  const gl = webglCanvas.getContext('webgl2', {
    alpha:              true,
    premultipliedAlpha: false,
    antialias:          false,
    depth:              false,
    stencil:            false,
  });

  if (!gl) {
    console.warn('[WebGL] WebGL2 not supported — Canvas2D overlay stays active');
    window._webglOverlayActive = false;
    window.__overlayRenderer = 'canvas2d';
    return;
  }

  // ── GLSL sources ──────────────────────────────────────────────────────

  // Common quad vertices used as gl_VertexID offsets for instanced quads.
  // Two triangles → one quad per instance (no index buffer needed).
  const _QUAD_CONST = `const vec2 QUAD[6]=vec2[6](
    vec2(-1.,-1.),vec2(1.,-1.),vec2(-1.,1.),
    vec2(1.,-1.), vec2(1.,1.), vec2(-1.,1.));`;

  const _QUAD_UV_CONST = `const vec2 UV[6]=vec2[6](
    vec2(0.,0.),vec2(1.,0.),vec2(0.,1.),
    vec2(1.,0.),vec2(1.,1.),vec2(0.,1.));`;

  // Shared fragment shader for all point programs.
  const FRAG_POINTS = `#version 300 es
precision mediump float;
in vec2 v_q;
out vec4 C;
void main(){
  float d=length(v_q);
  float a=1.0-smoothstep(0.55,1.0,d);
  if(a<0.01)discard;
  C=vec4(0.165,0.875,0.647,a*0.92);
}`;

  // rowWidth=3 data: [cx, cy, conf] stride=12
  const VERT_POINTS_CENTER = `#version 300 es
precision highp float;
layout(location=0) in vec2 a_c;   // cx,cy normalised
layout(location=1) in float a_w;  // conf (unused for shape)
uniform vec2 u_off,u_sz,u_cvs;
uniform float u_r;
${_QUAD_CONST}
out vec2 v_q;
void main(){
  v_q=QUAD[gl_VertexID];
  vec2 px=u_off+a_c*u_sz+v_q*u_r;
  vec2 c=px/u_cvs*2.0-1.0; c.y=-c.y;
  gl_Position=vec4(c,0.,1.);
}`;

  // rowWidth=5 data: [x1,y1,x2,y2,conf] stride=20, centres computed in shader
  const VERT_POINTS_BBOX = `#version 300 es
precision highp float;
layout(location=0) in vec4 a_b;   // x1,y1,x2,y2 normalised
layout(location=1) in float a_w;  // conf
uniform vec2 u_off,u_sz,u_cvs;
uniform float u_r;
${_QUAD_CONST}
out vec2 v_q;
void main(){
  v_q=QUAD[gl_VertexID];
  vec2 center=(a_b.xy+a_b.zw)*0.5;
  vec2 px=u_off+center*u_sz+v_q*u_r;
  vec2 c=px/u_cvs*2.0-1.0; c.y=-c.y;
  gl_Position=vec4(c,0.,1.);
}`;

  // rowWidth=5 data: [x1,y1,x2,y2,conf] stride=20, outlined rect
  const VERT_BBOX = `#version 300 es
precision highp float;
layout(location=0) in vec4 a_b;
layout(location=1) in float a_w;
uniform vec2 u_off,u_sz,u_cvs;
${_QUAD_UV_CONST}
out vec2 v_uv,v_bpx;
void main(){
  vec2 uv=UV[gl_VertexID];
  float px=u_off.x+mix(a_b.x,a_b.z,uv.x)*u_sz.x;
  float py=u_off.y+mix(a_b.y,a_b.w,uv.y)*u_sz.y;
  vec2 c=vec2(px,py)/u_cvs*2.0-1.0; c.y=-c.y;
  gl_Position=vec4(c,0.,1.);
  v_uv=uv;
  v_bpx=vec2((a_b.z-a_b.x)*u_sz.x,(a_b.w-a_b.y)*u_sz.y);
}`;

  const FRAG_BBOX = `#version 300 es
precision mediump float;
in vec2 v_uv,v_bpx;
out vec4 C;
uniform float u_bpx;
void main(){
  float ex=v_uv.x*v_bpx.x, ey=v_uv.y*v_bpx.y;
  bool edge=ex<u_bpx||ex>v_bpx.x-u_bpx||ey<u_bpx||ey>v_bpx.y-u_bpx;
  C=edge?vec4(0.165,0.875,0.647,0.9):vec4(0.165,0.875,0.647,0.10);
}`;

  // ── Shader helpers ────────────────────────────────────────────────────

  function _shader(type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src.trim());
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      console.error('[WebGL] Shader compile error:', gl.getShaderInfoLog(s));
      gl.deleteShader(s);
      return null;
    }
    return s;
  }

  function _prog(vsrc, fsrc) {
    const v = _shader(gl.VERTEX_SHADER, vsrc);
    const f = _shader(gl.FRAGMENT_SHADER, fsrc);
    if (!v || !f) { gl.deleteShader(v); gl.deleteShader(f); return null; }
    const p = gl.createProgram();
    gl.attachShader(p, v); gl.deleteShader(v);
    gl.attachShader(p, f); gl.deleteShader(f);
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      console.error('[WebGL] Link error:', gl.getProgramInfoLog(p));
      gl.deleteProgram(p);
      return null;
    }
    return p;
  }

  const progPC = _prog(VERT_POINTS_CENTER, FRAG_POINTS);  // centers data
  const progPB = _prog(VERT_POINTS_BBOX,   FRAG_POINTS);  // bbox→points
  const progBB = _prog(VERT_BBOX,          FRAG_BBOX);    // bbox draw

  if (!progPC || !progPB || !progBB) {
    window._webglOverlayActive = false;
    window.__overlayRenderer = 'canvas2d';
    console.error('[WebGL] Program compilation failed — Canvas2D fallback active');
    return;
  }

  // ── Uniform locations ─────────────────────────────────────────────────

  function _uloc(prog, names) {
    const u = {};
    for (const n of names) u[n] = gl.getUniformLocation(prog, n);
    return u;
  }

  const POINT_UNIFORMS = ['u_off', 'u_sz', 'u_cvs', 'u_r'];
  const uPC = _uloc(progPC, POINT_UNIFORMS);
  const uPB = _uloc(progPB, POINT_UNIFORMS);
  const uBB = _uloc(progBB, ['u_off', 'u_sz', 'u_cvs', 'u_bpx']);

  // ── VAOs + single shared VBO ──────────────────────────────────────────
  // vaoNarrow: reads stride=12 (centers: cx,cy,conf)
  // vaoWide:   reads stride=20 (bbox: x1,y1,x2,y2,conf)
  // Both share one VBO — only the attrib pointer layout differs.

  const vbo = gl.createBuffer();

  function _makeVao(stride, attrib0Components, attrib0Offset, attrib1Offset) {
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, attrib0Components, gl.FLOAT, false, stride, attrib0Offset);
    gl.vertexAttribDivisor(0, 1);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 1, gl.FLOAT, false, stride, attrib1Offset);
    gl.vertexAttribDivisor(1, 1);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.bindVertexArray(null);
    return vao;
  }

  // stride=12  attrib0=vec2@0  attrib1=float@8
  const vaoNarrow = _makeVao(12, 2, 0, 8);
  // stride=20  attrib0=vec4@0  attrib1=float@16
  const vaoWide   = _makeVao(20, 4, 0, 16);

  // ── Runtime state ────────────────────────────────────────────────────

  let _lastFrameId    = -2;
  let _lastRPts       = renderPoints;
  let _instanceCount  = 0;
  let _uploadedWidth  = 5;

  function _uploadVBO() {
    const rows     = _packedDetections;
    const rowWidth = _packedDetectionsRowWidth || 5;
    if (!rows || rows.length < rowWidth) { _instanceCount = 0; return; }
    _instanceCount = Math.floor(rows.length / rowWidth);
    _uploadedWidth = rowWidth;
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, rows, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  // ── rAF render loop ───────────────────────────────────────────────────

  function _render() {
    requestAnimationFrame(_render);

    // Sync canvas size to wrapper
    const wrapper = webglCanvas.parentElement;
    const W = wrapper.clientWidth  | 0;
    const H = wrapper.clientHeight | 0;
    if (webglCanvas.width !== W || webglCanvas.height !== H) {
      webglCanvas.width  = W;
      webglCanvas.height = H;
    }
    if (W === 0 || H === 0) return;

    gl.viewport(0, 0, W, H);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    // Always sync VBO so count stays current even when hidden
    const fid = _packedDetectionsFrameId;
    if (fid !== _lastFrameId || renderPoints !== _lastRPts) {
      _lastFrameId = fid;
      _lastRPts    = renderPoints;
      _uploadVBO();
    }

    if (!showMask || previewOnly || _instanceCount === 0) return;

    const layout = _computeLayout();
    if (!layout) return;

    const { dispX, dispY, dispW, dispH } = layout;
    const centersMode = (_uploadedWidth === 3);
    const usePoints   = centersMode || renderPoints;

    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(
      gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA,
      gl.ONE,       gl.ONE_MINUS_SRC_ALPHA,
    );

    if (usePoints) {
      const prog = centersMode ? progPC    : progPB;
      const vao  = centersMode ? vaoNarrow : vaoWide;
      const u    = centersMode ? uPC       : uPB;
      gl.useProgram(prog);
      gl.uniform2f(u.u_off, dispX, dispY);
      gl.uniform2f(u.u_sz,  dispW, dispH);
      gl.uniform2f(u.u_cvs, W, H);
      gl.uniform1f(u.u_r,   3.5);
      gl.bindVertexArray(vao);
      gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, _instanceCount);
    } else {
      gl.useProgram(progBB);
      gl.uniform2f(uBB.u_off,  dispX, dispY);
      gl.uniform2f(uBB.u_sz,   dispW, dispH);
      gl.uniform2f(uBB.u_cvs,  W, H);
      gl.uniform1f(uBB.u_bpx,  2.0);
      gl.bindVertexArray(vaoWide);
      gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, _instanceCount);
    }

    gl.bindVertexArray(null);
    gl.useProgram(null);
  }

  // Activate and start loop
  window._webglOverlayActive = true;
  window.__overlayRenderer = 'webgl2';
  webglCanvas.style.display  = '';
  console.info('[WebGL] WebGL2 instanced overlay renderer active');
  requestAnimationFrame(_render);
})();
