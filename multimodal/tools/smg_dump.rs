//! The SMG side of bellwether's multimodal suite: SMG's media pipeline over a list of image files, one JSON line per
//! image with the decoded size, the image grid, the pixel tensor's shape and the sha256 of the decoded RGB bytes, of the
//! u8 patches, of the bf16 wire bytes and of the f32 wire bytes, which `bellwether multimodal verify` compares with the
//! engine's own image path. Build it as an example of SMG's multimodal crate (`crates/multimodal/examples/`); it uses
//! the crate's public API alone, and SHA-256 is written out here so that the example needs no dependency the crate
//! does not have.
//!
//! Modes (`SMG_DUMP_MODE`): `hash` (default): one JSON line per image to the file `SMG_DUMP_OUT`: decoded width and
//! height, alpha flag, grid, shape, the four sha256 digests, and the wall time of decode, preprocess and the two writes.
//! `raw`: the artifacts per image into the directory `SMG_DUMP_OUT` (`<name>.{rgb,u8,bf16,f32}.raw` + `meta.json`),
//! for the first differing element. `time`: per image the median of `SMG_DUMP_REPEATS` runs of decode -> preprocess ->
//! bf16 write (and the f32 write), one JSON line each.
//! Env: `SMG_DUMP_LIST` (one path per line), `SMG_DUMP_PROCESSOR` (qwen2_vl | qwen3_vl | minimax_m3), `SMG_DUMP_CONFIG`
//! (the family's preprocessor_config.json), `SMG_DUMP_SYNTH` (optional comma list of WxH synthetic gradients),
//! `SMG_DUMP_REPEATS`.
#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::restriction,
    unused,
    unused_qualifications,
    unsafe_code,
    missing_docs
)]

use std::{
    env, fs,
    io::Write,
    panic::{catch_unwind, AssertUnwindSafe},
    path::PathBuf,
    time::Instant,
};

use image::{DynamicImage, RgbImage};
use llm_multimodal::{
    encoder_inputs::{EncoderDtype, EncoderInput, ModelSpecificValue},
    types::{ImageDetail, ImageFrame, ImageSize, ImageSource},
    vision::{
        preprocessor_config::PreProcessorConfig,
        processors::{MiniMaxM3VisionProcessor, Qwen2VLProcessor, Qwen3VLProcessor},
        VisionPreProcessor,
    },
};
use serde_json::json;

/// SHA-256 (FIPS 180-4) of `bytes`, as lowercase hex: the digest the suite's references carry.
fn sha(bytes: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
        0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
        0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
        0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];
    let mut data = bytes.to_vec();
    let bit_len = (bytes.len() as u64).wrapping_mul(8);
    data.push(0x80);
    while data.len() % 64 != 56 {
        data.push(0);
    }
    data.extend_from_slice(&bit_len.to_be_bytes());
    let mut w = [0u32; 64];
    for block in data.chunks_exact(64) {
        for (i, word) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (slot, value) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *slot = slot.wrapping_add(value);
        }
    }
    h.iter().map(|v| format!("{v:08x}")).collect()
}

fn make(w: u32, h: u32) -> DynamicImage {
    DynamicImage::ImageRgb8(RgbImage::from_fn(w, h, |x, y| {
        image::Rgb([
            ((x * 7 + y * 3) % 256) as u8,
            ((x * 5 + y * 11) % 256) as u8,
            ((x + y * 2) % 256) as u8,
        ])
    }))
}

fn header_size(bytes: &[u8]) -> Result<ImageSize, String> {
    let (w, h) = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| e.to_string())?
        .into_dimensions()
        .map_err(|e| e.to_string())?;
    Ok(ImageSize::new(w, h))
}

/// The gateway's decode of inline bytes (the frame's lazy `decode_image_bytes`: orientation applied, no cap).
fn decode(bytes: &[u8]) -> Result<DynamicImage, String> {
    let size = header_size(bytes).unwrap_or_else(|_| ImageSize::new(1, 1));
    let frame = ImageFrame::encoded(
        bytes::Bytes::copy_from_slice(bytes),
        ImageDetail::Auto,
        ImageSource::InlineBytes,
        String::new(),
        size,
        None,
    );
    frame.image().map(Clone::clone).map_err(|e| e.to_string())
}

struct Artifacts {
    w: u32,
    h: u32,
    alpha: bool,
    kind: &'static str,
    shape: Vec<usize>,
    grid: Vec<i64>,
    tokens: Vec<usize>,
    rgb: Vec<u8>,
    u8: Option<Vec<u8>>,
    bf16: Vec<u8>,
    f32: Vec<u8>,
    decode_ms: f64,
    pre_ms: f64,
    bf16_ms: f64,
    f32_ms: f64,
}

fn run(processor: &dyn VisionPreProcessor, config: &PreProcessorConfig, image: DynamicImage, decode_ms: f64) -> Result<Artifacts, String> {
    let alpha = image.color().has_alpha();
    let rgb = image.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let t0 = Instant::now();
    let pre = processor
        .preprocess(std::slice::from_ref(&image), config)
        .map_err(|e| e.to_string())?;
    let pre_ms = t0.elapsed().as_secs_f64() * 1e3;
    let shape = pre.encoder_input.shape().to_vec();
    let (kind, u8) = match &pre.encoder_input {
        EncoderInput::U8 { patches, .. } => {
            let std_layout = patches.as_standard_layout();
            ("u8", Some(std_layout.as_slice().unwrap().to_vec()))
        }
        EncoderInput::F32(_) => ("f32", None),
    };
    let t1 = Instant::now();
    let bf16 = pre.encoder_input.write_as(EncoderDtype::BFloat16);
    let bf16_ms = t1.elapsed().as_secs_f64() * 1e3;
    let t2 = Instant::now();
    let f32 = pre.encoder_input.write_as(EncoderDtype::Float32);
    let f32_ms = t2.elapsed().as_secs_f64() * 1e3;
    let grid = match pre.model_specific.get("image_grid_thw") {
        Some(ModelSpecificValue::IntTensor { data, .. }) => data.clone(),
        _ => vec![],
    };
    Ok(Artifacts {
        w,
        h,
        alpha,
        kind,
        shape,
        grid,
        tokens: pre.feature_token_counts.clone(),
        rgb: rgb.into_raw(),
        u8,
        bf16,
        f32,
        decode_ms,
        pre_ms,
        bf16_ms,
        f32_ms,
    })
}

fn main() {
    assert_eq!(
        sha(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "the SHA-256 implementation gives the FIPS 180-4 test vector"
    );
    let mode = env::var("SMG_DUMP_MODE").unwrap_or_else(|_| "hash".to_string());
    let out = PathBuf::from(env::var("SMG_DUMP_OUT").expect("SMG_DUMP_OUT"));
    let config_json = fs::read_to_string(env::var("SMG_DUMP_CONFIG").expect("SMG_DUMP_CONFIG")).unwrap();
    let config = PreProcessorConfig::from_json(&config_json).unwrap();
    let which = env::var("SMG_DUMP_PROCESSOR").unwrap_or_else(|_| "qwen3_vl".to_string());
    let processor: Box<dyn VisionPreProcessor> = match which.as_str() {
        "qwen2_vl" => Box::new(Qwen2VLProcessor::from_preprocessor_config(&config)),
        "qwen3_vl" => Box::new(Qwen3VLProcessor::from_preprocessor_config(&config)),
        "minimax_m3" => Box::new(MiniMaxM3VisionProcessor::from_preprocessor_config(&config).unwrap()),
        other => panic!("unknown processor {other}"),
    };
    let repeats: usize = env::var("SMG_DUMP_REPEATS").ok().and_then(|s| s.parse().ok()).unwrap_or(5);
    let mut items: Vec<(String, Option<PathBuf>, Option<(u32, u32)>)> = Vec::new();
    if let Ok(list) = env::var("SMG_DUMP_LIST") {
        for line in fs::read_to_string(&list).unwrap().lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let p = PathBuf::from(line);
            items.push((p.file_name().unwrap().to_string_lossy().to_string(), Some(p), None));
        }
    }
    for case in env::var("SMG_DUMP_SYNTH").unwrap_or_default().split(',').filter(|s| !s.is_empty()) {
        let (w, h) = case.split_once('x').unwrap();
        items.push((format!("synth_{w}x{h}"), None, Some((w.parse().unwrap(), h.parse().unwrap()))));
    }
    let mut meta: Vec<serde_json::Value> = Vec::new();
    let mut out_file = if mode == "raw" {
        fs::create_dir_all(&out).unwrap();
        None
    } else {
        Some(std::io::BufWriter::new(fs::File::create(&out).unwrap()))
    };
    let mut n_ok = 0usize;
    let mut n_err = 0usize;
    for (name, path, synth) in &items {
        let path_s = path.as_ref().map(|p| p.to_string_lossy().to_string()).unwrap_or_default();
        let result = catch_unwind(AssertUnwindSafe(|| -> Result<serde_json::Value, String> {
            let bytes = match path {
                Some(p) => fs::read(p).map_err(|e| e.to_string())?,
                None => Vec::new(),
            };
            let load = |_: ()| -> Result<(DynamicImage, f64), String> {
                let t = Instant::now();
                let img = match synth {
                    Some((w, h)) => make(*w, *h),
                    None => decode(&bytes)?,
                };
                Ok((img, t.elapsed().as_secs_f64() * 1e3))
            };
            if mode == "time" {
                let mut dec = Vec::new();
                let mut pre = Vec::new();
                let mut bf = Vec::new();
                let mut f3 = Vec::new();
                let mut last: Option<Artifacts> = None;
                for _ in 0..repeats {
                    let (img, d) = load(())?;
                    let a = run(processor.as_ref(), &config, img, d)?;
                    dec.push(a.decode_ms);
                    pre.push(a.pre_ms);
                    bf.push(a.bf16_ms);
                    f3.push(a.f32_ms);
                    last = Some(a);
                }
                let med = |v: &mut Vec<f64>| {
                    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    v[v.len() / 2]
                };
                let a = last.unwrap();
                return Ok(json!({"name": name, "path": path_s, "w": a.w, "h": a.h, "alpha": a.alpha, "kind": a.kind, "shape": a.shape,
                    "grid": a.grid, "repeats": repeats, "decode_ms": med(&mut dec), "pre_ms": med(&mut pre), "bf16_ms": med(&mut bf),
                    "f32_ms": med(&mut f3), "elements": a.bf16.len() / 2}));
            }
            let (img, d) = load(())?;
            let a = run(processor.as_ref(), &config, img, d)?;
            if mode == "raw" {
                fs::write(out.join(format!("{name}.rgb.raw")), &a.rgb).unwrap();
                if let Some(u8) = &a.u8 {
                    fs::write(out.join(format!("{name}.u8.raw")), u8).unwrap();
                }
                fs::write(out.join(format!("{name}.bf16.raw")), &a.bf16).unwrap();
                fs::write(out.join(format!("{name}.f32.raw")), &a.f32).unwrap();
                return Ok(json!({"name": name, "path": path_s, "kind": a.kind, "shape": a.shape, "rgb": [a.w, a.h], "alpha": a.alpha,
                    "grid": a.grid, "tokens": a.tokens}));
            }
            Ok(json!({"name": name, "path": path_s, "w": a.w, "h": a.h, "alpha": a.alpha, "kind": a.kind, "shape": a.shape, "grid": a.grid,
                "tokens": a.tokens, "rgb_sha256": sha(&a.rgb), "u8_sha256": a.u8.as_deref().map(sha), "bf16_sha256": sha(&a.bf16),
                "f32_sha256": sha(&a.f32), "decode_ms": a.decode_ms, "pre_ms": a.pre_ms, "bf16_ms": a.bf16_ms, "f32_ms": a.f32_ms}))
        }));
        let line = match result {
            Ok(Ok(v)) => {
                n_ok += 1;
                v
            }
            Ok(Err(e)) => {
                n_err += 1;
                json!({"name": name, "path": path_s, "error": e})
            }
            Err(p) => {
                n_err += 1;
                let msg = p
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "panic".to_string());
                json!({"name": name, "path": path_s, "error": format!("PANIC: {msg}")})
            }
        };
        match out_file.as_mut() {
            Some(f) => {
                writeln!(f, "{line}").unwrap();
            }
            None => meta.push(line),
        }
    }
    if let Some(mut f) = out_file {
        f.flush().unwrap();
    } else {
        fs::write(
            out.join("meta.json"),
            serde_json::to_string(&json!({"processor": which, "images": meta})).unwrap(),
        )
        .unwrap();
    }
    eprintln!("DUMP_DONE mode={mode} processor={which} ok={n_ok} err={n_err}");
}
