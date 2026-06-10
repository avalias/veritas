//! Tokenizer determinism (brief Phase 3.2): the vendored tokenizer.json is
//! pinned by hash and encodings are pinned by golden ids — any drift in
//! the file or the wrapping library fails loudly.

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/artifacts");

#[test]
fn tokenizer_file_hash_pinned() {
    use sha3::{Digest, Sha3_256};
    let bytes = std::fs::read(format!("{DIR}/tokenizer.json")).expect("vendored tokenizer");
    let h: [u8; 32] = Sha3_256::digest(&bytes).into();
    let hex: String = h.iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(
        hex,
        "427a6d44f980384d37f957696c4f7d191d0e3b3b7819bf4ee5c84634dd86a4a0",
        "tokenizer.json changed — consensus break"
    );
}

#[test]
fn encodings_golden() {
    let t = tokenizers::Tokenizer::from_file(format!("{DIR}/tokenizer.json")).unwrap();
    let cases: &[(&str, &[u32])] = &[
        ("Will it rain in Paris tomorrow?", &[9945, 432, 11174, 304, 12095, 16577, 30]),
        ("The answer is no.", &[785, 4226, 374, 902, 13]),
        ("yes", &[9693]),
    ];
    for (text, want) in cases {
        let got = t.encode(*text, false).unwrap().get_ids().to_vec();
        assert_eq!(&got, want, "{text:?}");
    }
}
