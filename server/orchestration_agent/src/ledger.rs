use std::fs::OpenOptions;
use std::io::Write;

pub fn append(entry: &[u8]) {
    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open("ledger.log")
        .unwrap();

    f.write_all(entry).unwrap();
    f.write_all(b"\n").unwrap();
}
