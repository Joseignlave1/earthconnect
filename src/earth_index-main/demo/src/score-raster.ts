import type { FrameState } from 'ol/Map.js';
import { atlasLevel } from './score-atlas.ts';
import type { ScoreAtlas } from './score-atlas.ts';

const CACHE_BYTES = 128 * 1024 * 1024;
type TextureLevel = { textures: WebGLTexture[]; bytes: number };

const vertex = `#version 300 es
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

function fragment(banks: number, bankSize: number) {
  return `#version 300 es
precision highp float;
precision highp int;
precision highp usampler2D;
precision highp usampler2DArray;
uniform usampler2D u_lookup;
${Array.from({ length: banks }, (_, i) => `uniform usampler2DArray u_scores${i};`).join('\n')}
uniform vec3 u_worldX, u_worldY;
uniform vec2 u_ratio, u_origin, u_span;
uniform float u_height, u_maximum, u_cutoff;
uniform vec4 u_projection;
uniform bool u_sinusoidal;
out vec4 color;
void main() {
  vec3 pixel = vec3(gl_FragCoord.x / u_ratio.x, u_height - gl_FragCoord.y / u_ratio.y, 1.0);
  vec2 world = vec2(dot(u_worldX, pixel), dot(u_worldY, pixel));
  color = vec4(0.0);
  if (abs(world.x) > 180.0 || abs(world.y) > 90.0 || u_maximum <= 0.0) return;
  vec2 point = world;
  if (u_sinusoidal) {
    vec2 radians = world * 0.017453292519943295;
    point = vec2(u_projection.x * (radians.x - u_projection.y) * cos(radians.y) + u_projection.z,
                 u_projection.x * radians.y + u_projection.w);
  }
  vec2 address = vec2((point.x - u_origin.x) / u_span.x, (u_origin.y - point.y) / u_span.y);
  ivec2 region = ivec2(floor(address));
  if (any(lessThan(address, vec2(0.0))) || any(greaterThanEqual(region, textureSize(u_lookup, 0)))) return;
  uint slot = texelFetch(u_lookup, region, 0).r;
  if (slot == 65535u) return;
  int bank = int(slot) / ${bankSize};
  int plane = int(slot) % ${bankSize};
  uint score = 65535u;
  ${Array.from(
    { length: banks },
    (_, i) => `
  ${i ? 'else ' : ''}if (bank == ${i}) {
    ivec2 cell = ivec2(fract(address) * vec2(textureSize(u_scores${i}, 0).xy));
    score = texelFetch(u_scores${i}, ivec3(cell, plane), 0).r;
  }`,
  ).join('')}
  if (score == 65535u || float(score) < u_cutoff || score == 0u) return;
  float alpha = (235.0 / 255.0) * pow(min(1.0, float(score) / u_maximum), 1.6);
  color = vec4(vec3(168.0, 85.0, 247.0) / 255.0 * alpha, alpha);
}`;
}

// One canvas and one fullscreen draw. Scores stay lossless uint16 in integer
// texture arrays. LOD textures are built once on demand and retained in a
// bounded LRU; cutoff/normalization are uniforms, not baked image variants.
export class ScoreRaster {
  readonly element = document.createElement('div');
  readonly canvas = document.createElement('canvas');
  onRestore = () => {};
  private gl: WebGL2RenderingContext | null = null;
  private atlas: ScoreAtlas | null = null;
  private program: WebGLProgram | null = null;
  private lookup: WebGLTexture | null = null;
  private levels = new Map<number, TextureLevel>();
  private uniforms = new Map<string, WebGLUniformLocation | null>();
  private bankSize = 0;
  private lost = false;
  private uploads = 0;
  private bytes = 0;
  private draws = 0;
  constructor(private readonly budget = CACHE_BYTES) {
    this.element.className = 'ol-layer score-raster';
    Object.assign(this.element.style, { position: 'absolute', inset: '0', pointerEvents: 'none' });
    this.element.appendChild(this.canvas);
    this.canvas.setAttribute('aria-hidden', 'true');
    this.canvas.addEventListener('webglcontextlost', this.contextLost);
    this.canvas.addEventListener('webglcontextrestored', this.contextRestored);
  }
  get stats() {
    return {
      uploads: this.uploads,
      cacheBytes: this.bytes,
      cachedLevels: this.levels.size,
      draws: this.draws,
    };
  }
  private contextLost = (event: Event) => {
    event.preventDefault();
    this.lost = true;
    this.program = null;
    this.lookup = null;
    this.levels.clear();
    this.uniforms.clear();
    this.bytes = 0;
  };
  private contextRestored = () => {
    this.lost = false;
    if (this.atlas) this.initialize();
    this.onRestore();
  };
  set(atlas: ScoreAtlas) {
    const native = atlas.patches[0].levels[0];
    if (native.width * native.height * atlas.patches.length * 2 > this.budget)
      throw new Error('Score grids exceed the GPU score cache budget');
    this.clear();
    this.atlas = atlas;
    this.initialize();
  }
  private initialize() {
    const gl = (this.gl ??= this.canvas.getContext('webgl2', {
      alpha: true,
      premultipliedAlpha: true,
    }));
    if (!gl) throw new Error('WebGL2 is required to display score grids');
    if (this.lost || !this.atlas) return;
    this.bankSize = gl.getParameter(gl.MAX_ARRAY_TEXTURE_LAYERS) as number;
    const banks = Math.ceil(this.atlas.patches.length / this.bankSize);
    if (banks + 1 > (gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS) as number))
      throw new Error('Too many score regions for this GPU');
    const shape = this.atlas.patches[0].levels[0];
    if (
      Math.max(shape.width, shape.height, this.atlas.columns, this.atlas.rows) >
      (gl.getParameter(gl.MAX_TEXTURE_SIZE) as number)
    )
      throw new Error('Score grid exceeds this GPU texture size');
    const shader = (type: number, source: string) => {
      const object = gl.createShader(type)!;
      gl.shaderSource(object, source);
      gl.compileShader(object);
      if (!gl.getShaderParameter(object, gl.COMPILE_STATUS)) {
        const error = gl.getShaderInfoLog(object);
        gl.deleteShader(object);
        throw new Error(`Score shader: ${error}`);
      }
      return object;
    };
    const vs = shader(gl.VERTEX_SHADER, vertex);
    const fs = shader(gl.FRAGMENT_SHADER, fragment(banks, this.bankSize));
    const program = gl.createProgram()!;
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const error = gl.getProgramInfoLog(program);
      gl.deleteProgram(program);
      throw new Error(`Score shader: ${error}`);
    }
    this.program = program;
    this.lookup = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.lookup);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.R16UI,
      this.atlas.columns,
      this.atlas.rows,
      0,
      gl.RED_INTEGER,
      gl.UNSIGNED_SHORT,
      this.atlas.lookup,
    );
    this.checkError();
  }
  private checkError() {
    const error = this.gl!.getError();
    if (error !== this.gl!.NO_ERROR) throw new Error(`Could not upload score textures (${error})`);
  }
  private uniform(name: string) {
    if (!this.uniforms.has(name))
      this.uniforms.set(name, this.gl!.getUniformLocation(this.program!, name));
    return this.uniforms.get(name)!;
  }
  private upload(level: number): TextureLevel {
    const hit = this.levels.get(level);
    if (hit) {
      this.levels.delete(level);
      this.levels.set(level, hit);
      return hit;
    }
    const gl = this.gl!,
      atlas = this.atlas!;
    const shape = atlas.patches[0].levels[level];
    const bytes = shape.width * shape.height * atlas.patches.length * 2;
    while (this.bytes + bytes > this.budget && this.levels.size) {
      const [key, old] = this.levels.entries().next().value!;
      for (const texture of old.textures) gl.deleteTexture(texture);
      this.levels.delete(key);
      this.bytes -= old.bytes;
    }
    const textures: WebGLTexture[] = [];
    try {
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      for (let start = 0; start < atlas.patches.length; start += this.bankSize) {
        const count = Math.min(this.bankSize, atlas.patches.length - start);
        const texture = gl.createTexture()!;
        textures.push(texture);
        gl.bindTexture(gl.TEXTURE_2D_ARRAY, texture);
        gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texStorage3D(gl.TEXTURE_2D_ARRAY, 1, gl.R16UI, shape.width, shape.height, count);
        for (let i = 0; i < count; i++)
          gl.texSubImage3D(
            gl.TEXTURE_2D_ARRAY,
            0,
            0,
            0,
            i,
            shape.width,
            shape.height,
            1,
            gl.RED_INTEGER,
            gl.UNSIGNED_SHORT,
            atlas.patches[start + i].levels[level].scores,
          );
        this.checkError();
      }
    } catch (error) {
      for (const texture of textures) gl.deleteTexture(texture);
      throw error;
    }
    const result = { textures, bytes };
    this.uploads++;
    this.bytes += bytes;
    this.levels.set(level, result);
    return result;
  }
  render(frame: FrameState, cutoff: number, maximum: number) {
    if (!this.atlas || this.lost || !this.program) return this.element;
    const gl = this.gl!;
    const width = Math.round(frame.size[0] * frame.pixelRatio),
      height = Math.round(frame.size[1] * frame.pixelRatio);
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
      this.canvas.style.width = `${frame.size[0]}px`;
      this.canvas.style.height = `${frame.size[1]}px`;
    }
    gl.viewport(0, 0, width, height);
    gl.useProgram(this.program);
    const level = this.upload(atlasLevel(this.atlas, frame.viewState.resolution));
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.lookup);
    gl.uniform1i(this.uniform('u_lookup'), 0);
    level.textures.forEach((texture, i) => {
      gl.activeTexture(gl.TEXTURE0 + i + 1);
      gl.bindTexture(gl.TEXTURE_2D_ARRAY, texture);
      gl.uniform1i(this.uniform(`u_scores${i}`), i + 1);
    });
    const t = frame.pixelToCoordinateTransform,
      p = this.atlas.projection;
    gl.uniform3f(this.uniform('u_worldX'), t[0], t[2], t[4]);
    gl.uniform3f(this.uniform('u_worldY'), t[1], t[3], t[5]);
    gl.uniform2f(this.uniform('u_ratio'), width / frame.size[0], height / frame.size[1]);
    gl.uniform1f(this.uniform('u_height'), frame.size[1]);
    gl.uniform2f(this.uniform('u_origin'), ...(this.atlas.origin as [number, number]));
    gl.uniform2f(this.uniform('u_span'), ...(this.atlas.span as [number, number]));
    gl.uniform4f(this.uniform('u_projection'), p.radius, p.longitude, p.x, p.y);
    gl.uniform1i(this.uniform('u_sinusoidal'), p.sinusoidal ? 1 : 0);
    gl.uniform1f(this.uniform('u_maximum'), maximum);
    gl.uniform1f(this.uniform('u_cutoff'), cutoff);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    this.draws++;
    return this.element;
  }
  clear() {
    const gl = this.gl;
    if (gl && !this.lost) {
      for (const level of this.levels.values())
        for (const texture of level.textures) gl.deleteTexture(texture);
      if (this.lookup) gl.deleteTexture(this.lookup);
      if (this.program) gl.deleteProgram(this.program);
    }
    this.levels.clear();
    this.uniforms.clear();
    this.lookup = null;
    this.program = null;
    this.atlas = null;
    this.bytes = 0;
    this.uploads = 0;
  }
  destroy() {
    this.clear();
    this.canvas.removeEventListener('webglcontextlost', this.contextLost);
    this.canvas.removeEventListener('webglcontextrestored', this.contextRestored);
    this.gl?.getExtension('WEBGL_lose_context')?.loseContext();
    this.gl = null;
  }
}
