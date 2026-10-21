#import bevy_ui::ui_vertex_output::UiVertexOutput

@group(1) @binding(0) var<uniform> optics: vec4<f32>;
@group(1) @binding(1) var<uniform> impact: vec4<f32>;

fn segment(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>, width: f32, aa: f32) -> f32 {
    let delta = b - a;
    let t = clamp(dot(p - a, delta) / max(dot(delta, delta), 0.0000001), 0.0, 1.0);
    return 1.0 - smoothstep(width, width + aa, length(p - a - t * delta));
}

@fragment
fn fragment(in: UiVertexOutput) -> @location(0) vec4<f32> {
    // Height-normalized coordinates keep the aperture circular in any aspect ratio.
    let aspect = in.size.x / in.size.y;
    let pixel = 1.0 / in.size.y;
    let p = (in.uv - vec2<f32>(0.5)) * vec2<f32>(aspect, 1.0);
    let r = length(p);
    let radius = optics.z;
    if r > radius {
        let rim = 1.0 - smoothstep(0.0, 0.018, r - radius);
        let bevel = exp(-pow((r - radius - 0.009) / 0.0013, 2.0));
        let lip = exp(-pow((r - radius - 0.0018) / 0.0008, 2.0));
        let metal = 0.008 + rim * 0.016 + bevel * 0.024 + lip * 0.045;
        return vec4<f32>(vec3<f32>(metal * 0.94, metal, metal * 1.02), 0.995);
    }
    let vignette = smoothstep(radius * 0.71, radius, r);
    var alpha = vignette * vignette * 0.74;
    var rgb = vec3<f32>(0.006, 0.009, 0.008);
    // Thin etched centre and heavier duplex posts, with a dark keyline for bright scenes.
    let horizontal = segment(p, vec2<f32>(-radius * 0.9, 0.0), vec2<f32>(radius * 0.9, 0.0), 0.28 * pixel, 0.65 * pixel);
    let vertical = segment(p, vec2<f32>(0.0, -radius * 0.9), vec2<f32>(0.0, radius * 0.84), 0.28 * pixel, 0.65 * pixel);
    var line = max(horizontal, vertical);
    let post_a = segment(p, vec2<f32>(-radius * 0.9, 0.0), vec2<f32>(-radius * 0.56, 0.0), 1.05 * pixel, 0.65 * pixel);
    let post_b = segment(p, vec2<f32>(radius * 0.56, 0.0), vec2<f32>(radius * 0.9, 0.0), 1.05 * pixel, 0.65 * pixel);
    let post_c = segment(p, vec2<f32>(0.0, -radius * 0.9), vec2<f32>(0.0, -radius * 0.56), 1.05 * pixel, 0.65 * pixel);
    line = max(line, max(post_a, max(post_b, post_c)));
    // Actual angular divisions: each small division is ten milliradians.
    if abs(p.y) < 9.0 * pixel {
        for (var i = 1; i <= 10; i = i + 1) {
            let x = optics.x * tan(f32(i) * 0.01);
            let height = select(3.0, 5.0, i % 2 == 0) * pixel;
            line = max(line, segment(p, vec2<f32>(x, -height), vec2<f32>(x, height), 0.25 * pixel, 0.6 * pixel));
            line = max(line, segment(p, vec2<f32>(-x, -height), vec2<f32>(-x, height), 0.25 * pixel, 0.6 * pixel));
        }
    }
    // A dim amber centre illumination, without a floating game-style target marker.
    let centre = 1.0 - smoothstep(0.65 * pixel, 1.4 * pixel, r);
    let etched = vec3<f32>(0.29, 0.28, 0.23);
    rgb = mix(rgb, etched, line * 0.82);
    alpha = max(alpha, line * 0.82);
    rgb = mix(rgb, vec3<f32>(0.43, 0.28, 0.12), centre);
    alpha = max(alpha, centre * 0.88);
    // Current distance's computed ballistic hold mark is a small chevron on the reticle.
    if impact.z > 0.5 {
        let q = p - (impact.xy - vec2<f32>(0.5)) * vec2<f32>(aspect, 1.0);
        let left = segment(q, vec2<f32>(-7.0, 3.0) * pixel, vec2<f32>(0.0), 0.5 * pixel, 0.65 * pixel);
        let right = segment(q, vec2<f32>(7.0, 3.0) * pixel, vec2<f32>(0.0), 0.5 * pixel, 0.65 * pixel);
        let hold = max(left, right);
        rgb = mix(rgb, vec3<f32>(0.43, 0.33, 0.17), hold);
        alpha = max(alpha, hold * 0.9);
    }
    return vec4<f32>(rgb, alpha);
}
