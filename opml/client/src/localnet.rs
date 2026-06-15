//! Thin wrapper over the `sui` CLI (--json) — boring and auditable; the
//! heavyweight Rust SDK is not worth its dependency tree for a driver.

use serde_json::Value;
use std::process::Command;

pub struct Cli {
    pub package: String,
    active: std::cell::RefCell<String>,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct GasUsed {
    pub computation: u64,
    pub storage: u64,
    pub rebate: u64,
}

impl Cli {
    /// Queries the REAL active address — never trust a stale assumption
    /// (a previous run may have left the CLI switched elsewhere).
    pub fn new(package: String) -> Self {
        let out = Command::new("sui")
            .args(["client", "active-address"])
            .output()
            .expect("spawn sui");
        let active = String::from_utf8_lossy(&out.stdout).trim().to_string();
        Self { package, active: std::cell::RefCell::new(active) }
    }

    fn run(args: &[&str]) -> Result<Value, String> {
        let out = Command::new("sui")
            .args(args)
            .output()
            .map_err(|e| format!("spawn sui: {e}"))?;
        let stdout = String::from_utf8_lossy(&out.stdout);
        if !out.status.success() {
            return Err(format!(
                "sui {:?} failed:\n{}\n{}",
                &args[..args.len().min(6)],
                stdout,
                String::from_utf8_lossy(&out.stderr)
            ));
        }
        // Some subcommands print non-JSON preamble lines; find the JSON.
        let start = stdout.find(['{', '[']).ok_or_else(|| format!("no JSON in: {stdout}"))?;
        serde_json::from_str(&stdout[start..]).map_err(|e| format!("parse: {e}\n{stdout}"))
    }

    pub fn switch(&self, addr: &str) {
        if *self.active.borrow() == addr {
            return;
        }
        let out = Command::new("sui")
            .args(["client", "switch", "--address", addr])
            .output()
            .expect("spawn sui");
        assert!(out.status.success(), "switch failed");
        *self.active.borrow_mut() = addr.to_string();
    }

    /// Richest gas coin owned by the active address (earlier splits leave
    /// small change behind; always fund bonds from the largest).
    pub fn a_gas_coin(&self) -> String {
        let v = Self::run(&["client", "gas", "--json"]).expect("gas");
        let balance = |c: &Value| -> u64 {
            for k in ["mistBalance", "gasBalance", "balance"] {
                if let Some(n) = c[k].as_u64() {
                    return n;
                }
                if let Some(s) = c[k].as_str() {
                    if let Ok(n) = s.parse() {
                        return n;
                    }
                }
            }
            0
        };
        v.as_array()
            .expect("gas array")
            .iter()
            .max_by_key(|c| balance(c))
            .and_then(|c| c["gasCoinId"].as_str())
            .expect("gasCoinId")
            .to_string()
    }

    /// Split an exact bond off a gas coin; returns the new coin id.
    pub fn split_bond(&self, amount: u64) -> String {
        let coin = self.a_gas_coin();
        let v = Self::run(&[
            "client",
            "split-coin",
            "--coin-id",
            &coin,
            "--amounts",
            &amount.to_string(),
            "--gas-budget",
            "100000000",
            "--json",
        ])
        .expect("split");
        created_of_type(&v, "::coin::Coin").expect("split coin id")
    }

    /// Entry call on dispute::dispute as `sender`. Returns the full tx JSON.
    pub fn call(&self, sender: &str, function: &str, args: &[String]) -> Result<Value, String> {
        self.switch(sender);
        let mut argv: Vec<&str> = vec![
            "client",
            "call",
            "--package",
            &self.package,
            "--module",
            "dispute",
            "--function",
            function,
            "--gas-budget",
            "2000000000",
            "--json",
        ];
        if !args.is_empty() {
            argv.push("--args");
            for a in args {
                argv.push(a);
            }
        }
        let v = Self::run(&argv)?;
        let status = v["effects"]["status"]["status"].as_str().unwrap_or("?");
        if status != "success" {
            return Err(format!("{function}: {}", v["effects"]["status"]));
        }
        Ok(v)
    }

    pub fn object_fields(&self, id: &str) -> Value {
        let v = Self::run(&["client", "object", id, "--json"]).expect("object");
        v["content"]["fields"].clone()
    }
}

pub fn created_of_type(tx: &Value, type_frag: &str) -> Option<String> {
    tx["objectChanges"].as_array()?.iter().find_map(|c| {
        let t = c["objectType"].as_str()?;
        if c["type"].as_str() == Some("created") && t.contains(type_frag) {
            Some(c["objectId"].as_str()?.to_string())
        } else {
            None
        }
    })
}

pub fn gas_used(tx: &Value) -> GasUsed {
    let g = &tx["effects"]["gasUsed"];
    let n = |k: &str| {
        g[k].as_str()
            .and_then(|s| s.parse().ok())
            .or_else(|| g[k].as_u64())
            .unwrap_or(0)
    };
    GasUsed {
        computation: n("computationCost"),
        storage: n("storageCost"),
        rebate: n("storageRebate"),
    }
}

pub fn hex_arg(bytes: &[u8]) -> String {
    if bytes.is_empty() {
        // SuiJson: empty vector<u8>.
        "[]".to_string()
    } else {
        let mut s = String::with_capacity(2 + 2 * bytes.len());
        s.push_str("0x");
        for b in bytes {
            s.push_str(&format!("{b:02x}"));
        }
        s
    }
}

pub fn hex_vec_arg<T: AsRef<[u8]>>(items: &[T]) -> String {
    let inner: Vec<String> =
        items.iter().map(|h| format!("\"{}\"", hex_arg(h.as_ref()))).collect();
    format!("[{}]", inner.join(","))
}
