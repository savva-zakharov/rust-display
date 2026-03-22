use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

pub struct Lut {
    pub size: u32,
    pub data: Vec<f32>, // Flat R, G, B values
}

impl Lut {
    pub fn load_cube<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let file = File::open(&path).map_err(|e| e.to_string())?;
        let reader = BufReader::new(file);

        let mut size = 0;
        let mut data = Vec::new();

        for line in reader.lines() {
            let line = line.map_err(|e| e.to_string())?;
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            if line.starts_with("LUT_3D_SIZE") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    size = parts[1].parse().map_err(|_| "Invalid LUT size")?;
                }
                continue;
            }

            // Skip other keywords for now
            if line.starts_with("TITLE") || line.starts_with("DOMAIN_") || line.starts_with("LUT_1D_SIZE") {
                continue;
            }

            // Parse R G B
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 3 {
                for part in parts {
                    data.push(part.parse().map_err(|_| "Invalid LUT data")?);
                }
            }
        }

        if size == 0 {
            return Err("No LUT_3D_SIZE found".to_string());
        }

        if data.len() != (size * size * size * 3) as usize {
            return Err(format!("LUT data size mismatch: expected {}, got {}", size * size * size * 3, data.len()));
        }

        Ok(Lut { size, data })
    }
}

pub fn list_luts() -> Vec<PathBuf> {
    let mut luts = Vec::new();
    let luts_dir = Path::new("luts");
    if let Ok(entries) = std::fs::read_dir(luts_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "cube") {
                luts.push(path);
            }
        }
    }
    luts.sort();
    luts
}
