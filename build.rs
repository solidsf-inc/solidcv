use std::env;
use std::fs;
use std::path::PathBuf;

/// Derive the 64-bit Zobrist-Feistel master key from the 8-byte `solidSF` Schur-Feshbach invariant.
fn derive_zobrist_master_key() -> u64 {
    let tag: [u8; 8] = [0x53, 0x4F, 0x4C, 0x49, 0x44, 0x53, 0x46, 0x21];
    u64::from_be_bytes(tag)
}

#[inline]
fn keystream_byte(master: u64, idx: usize) -> u8 {
    let mut z = master ^ ((idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z & 0xFF) as u8
}

fn read_u32_le(buf: &[u8], cursor: &mut usize) -> u32 {
    let b: [u8; 4] = buf[*cursor..*cursor + 4].try_into().expect("vault u32");
    *cursor += 4;
    u32::from_le_bytes(b)
}

fn main() {
    println!("cargo:rerun-if-changed=assets/solidcv_moat_vault.bin");
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let vault_path = manifest_dir.join("assets").join("solidcv_moat_vault.bin");

    let enc = fs::read(&vault_path).expect("failed to read assets/solidcv_moat_vault.bin");
    let master = derive_zobrist_master_key();
    let dec: Vec<u8> = enc
        .iter()
        .enumerate()
        .map(|(i, &b)| b ^ keystream_byte(master, i))
        .collect();

    assert!(
        dec.starts_with(b"SSFVLT01"),
        "invalid solidcv_moat_vault.bin magic header"
    );
    let mut cur = 8usize;
    let count = read_u32_le(&dec, &mut cur) as usize;
    for _ in 0..count {
        let name_len = read_u32_le(&dec, &mut cur) as usize;
        let name = std::str::from_utf8(&dec[cur..cur + name_len]).expect("vault module name");
        cur += name_len;
        let body_len = read_u32_le(&dec, &mut cur) as usize;
        let body = &dec[cur..cur + body_len];
        cur += body_len;

        let target = out_dir.join(format!("vault_{name}.rs"));
        fs::write(&target, body).expect("failed to write unpacked vault module");
    }
}
