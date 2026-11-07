//! The home page (`GET /`): what the server is and the records it has of each game.

use crate::regions::Regions;
use crate::store::{natural_key, PartStats};
use crate::validate;
use std::collections::HashMap;
use std::fmt::Write;

/// The built-in game names (`TITLE=Name`).
pub const TITLES: &str = include_str!("titles.txt");

/// The names of the games by title.
pub fn parse_titles(text: &str) -> HashMap<String, String> {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.starts_with('#'))
        .filter_map(|l| l.split_once('='))
        .map(|(t, n)| (t.trim().to_string(), n.trim().to_string()))
        .filter(|(t, n)| validate::valid_title(t) && !n.is_empty())
        .collect()
}

/// What the page says besides the figures.
#[derive(Clone, Debug)]
pub struct Page {
    /// `owner/name` of the repository on GitHub.
    pub repo: String,
    /// Game names by title (the built-in table); then the regions table; then the title alone.
    pub titles: HashMap<String, String>,
    pub regions: Regions,
    /// The release the figures are compared with, and a note when it is not the latest release.
    pub release: Option<String>,
    pub note: Option<String>,
}

impl Default for Page {
    fn default() -> Self {
        Self {
            repo: "TheCruZ/KytyPS5-GTA".into(),
            titles: parse_titles(TITLES),
            regions: Regions::embedded(),
            release: None,
            note: None,
        }
    }
}

/// Text for HTML (element content and quoted attribute values).
pub fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    out
}

/// 12345 as "12,345".
fn num(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// `2026-11-05 20:14 UTC` from Unix seconds.
pub fn utc(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rest = secs % 86_400;
    // Civil date from days since 1970-01-01 (proleptic Gregorian calendar).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02} {:02}:{:02} UTC", rest / 3600, rest % 3600 / 60)
}

const STYLE: &str = "\
:root{color-scheme:light dark;--bg:#ffffff;--fg:#1c1e21;--muted:#5f6670;--line:#dfe3e8;--head:#f2f4f7;--link:#1f5fbf}\
@media (prefers-color-scheme:dark){:root{--bg:#15171b;--fg:#e5e7ea;--muted:#9aa2ac;--line:#2d323a;--head:#1e2228;--link:#82aefc}}\
*{box-sizing:border-box}\
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,-apple-system,\"Segoe UI\",Roboto,sans-serif}\
main{max-width:68rem;margin:0 auto;padding:1.5rem 1rem 3rem}\
h1{font-size:1.6rem;line-height:1.25;margin:.5rem 0 1rem}\
h2{font-size:1.2rem;margin:2rem 0 .5rem}\
p{margin:.6rem 0}\
a{color:var(--link)}\
.muted{color:var(--muted);font-size:.9rem}\
.wrap{overflow-x:auto;-webkit-overflow-scrolling:touch;border:1px solid var(--line);border-radius:6px}\
table{border-collapse:collapse;width:100%;font-size:.9rem;font-variant-numeric:tabular-nums}\
th,td{padding:.4rem .6rem;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}\
thead th{background:var(--head);font-weight:600}\
th.l,td.l{text-align:left}\
tr.game th{text-align:left;background:var(--head);font-weight:600;padding-top:.6rem}\
tfoot td{font-weight:600;border-bottom:0}\
";

/// The page, for the figures of the store at `now` (Unix seconds).
pub fn render(page: &Page, stats: &[PartStats], now: u64) -> String {
    let name = |title: &str| -> String {
        match page.titles.get(title) {
            Some(n) => n.clone(),
            None if page.regions.game_name(title) != title => page.regions.game_name(title).to_string(),
            None => title.to_string(),
        }
    };
    // The test partitions of the server (TEST00001...) are not games.
    let mut rows: Vec<(String, &PartStats)> =
        stats.iter().filter(|s| !s.key.title.starts_with("TEST")).map(|s| (name(&s.key.title), s)).collect();
    rows.sort_by(|a, b| {
        (a.0.to_lowercase(), &a.1.key.title, natural_key(&a.1.key.version), &a.1.key.format).cmp(&(
            b.0.to_lowercase(),
            &b.1.key.title,
            natural_key(&b.1.key.version),
            &b.1.key.format,
        ))
    });
    let release = page.release.as_deref().unwrap_or("release");
    let repo = escape(&page.repo);
    let mut h = String::new();
    let _ = write!(
        h,
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n\
         <meta name=\"color-scheme\" content=\"light dark\">\n\
         <title>KytyPS5-GTA pipeline sharing</title>\n<style>{STYLE}</style>\n</head>\n<body>\n<main>\n\
         <h1>Welcome to the KytyPS5-GTA pipeline sharing server</h1>\n\
         <p>This server shares the pipeline manifests of <a href=\"https://github.com/{repo}\">KytyPS5-GTA</a> \
         between the players who turn on <em>Share pipeline data</em> in the launcher: what the emulator of one \
         player discovers, the others receive. Knowing the shaders and pipelines of a game in advance, the \
         emulator prepares them while the game loads instead of in the middle of a scene, so games load faster \
         and stutter less.</p>\n\
         <p>It only keeps hashes of shaders and the state of pipelines, never code of the games. Regions and \
         versions of a game that run the same shaders share what they have.</p>\n\
         <p>Get the emulator from the <a href=\"https://github.com/{repo}/releases\">releases</a> of \
         <a href=\"https://github.com/{repo}\">{repo}</a>.</p>\n\
         <h2>Games</h2>\n"
    );
    let release_h = escape(release);
    let _ = write!(
        h,
        "<p class=\"muted\">Shaders are the distinct shaders recorded for a version of a game; pipelines, the \
         pipelines recorded with them. <em>In {release_h}</em> is what the release already ships; <em>From \
         players</em> is what players sent since (and what the repository gained after the release). \
         <em>From other regions and versions</em> is what a version also receives from compatible ones.</p>\n\
         <div class=\"wrap\"><table>\n<thead>\n\
         <tr><th class=\"l\" rowspan=\"2\" scope=\"col\">Title</th><th class=\"l\" rowspan=\"2\" scope=\"col\">Version</th>\
         <th colspan=\"3\" scope=\"colgroup\">Shaders</th><th colspan=\"3\" scope=\"colgroup\">Pipelines</th>\
         <th colspan=\"2\" scope=\"colgroup\">From other regions and versions</th></tr>\n\
         <tr><th scope=\"col\">Total</th><th scope=\"col\">In {release_h}</th><th scope=\"col\">From players</th>\
         <th scope=\"col\">Total</th><th scope=\"col\">In {release_h}</th><th scope=\"col\">From players</th>\
         <th scope=\"col\">Shaders</th><th scope=\"col\">Pipelines</th></tr>\n</thead>\n<tbody>\n"
    );
    let mut total = [0usize; 8];
    let mut game: Option<&str> = None;
    for (name, s) in &rows {
        if game != Some(name.as_str()) {
            let _ = writeln!(h, "<tr class=\"game\"><th colspan=\"10\" scope=\"colgroup\">{}</th></tr>", escape(name));
            game = Some(name.as_str());
        }
        let mut version = escape(&s.key.version);
        let format = validate::format_dir(&s.key.format);
        if format != "v1" {
            let _ = write!(version, " ({})", escape(&format));
        }
        let cells = [
            s.shaders,
            s.shaders_in_release,
            s.shaders - s.shaders_in_release,
            s.pipelines,
            s.pipelines_in_release,
            s.pipelines - s.pipelines_in_release,
            s.copied_shaders,
            s.copied_pipelines,
        ];
        let _ = write!(h, "<tr><td class=\"l\">{}</td><td class=\"l\">{version}</td>", escape(&s.key.title));
        for (i, c) in cells.iter().enumerate() {
            total[i] += c;
            let _ = write!(h, "<td>{}</td>", num(*c));
        }
        h.push_str("</tr>\n");
    }
    if rows.is_empty() {
        h.push_str("<tr><td class=\"l\" colspan=\"10\">Nothing recorded yet.</td></tr>\n");
    }
    let games = {
        let mut names: Vec<&str> = rows.iter().map(|r| r.0.as_str()).collect();
        names.dedup();
        names.len()
    };
    let _ = write!(
        h,
        "</tbody>\n<tfoot><tr><td class=\"l\">Total</td><td class=\"l\">{} games, {} versions</td>",
        num(games),
        num(rows.len())
    );
    for c in total {
        let _ = write!(h, "<td>{}</td>", num(c));
    }
    h.push_str("</tr></tfoot>\n</table></div>\n");
    let compared = match &page.release {
        Some(r) => format!("compared with the release {}", escape(r)),
        None => "with no release to compare with".to_string(),
    };
    let _ = writeln!(h, "<p class=\"muted\">Figures as of {}, {compared}.</p>", utc(now));
    if let Some(note) = &page.note {
        let _ = writeln!(h, "<p class=\"muted\">{}</p>", escape(note));
    }
    h.push_str("</main>\n</body>\n</html>\n");
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::PartKey;

    const FMT: &str = "# kyty pipeline manifest v1";

    fn stats(title: &str, version: &str, n: [usize; 6]) -> PartStats {
        PartStats {
            key: PartKey::new(FMT, title, version),
            shaders: n[0],
            shaders_in_release: n[1],
            pipelines: n[2],
            pipelines_in_release: n[3],
            copied_shaders: n[4],
            copied_pipelines: n[5],
        }
    }

    #[test]
    fn escapes_numbers_and_dates() {
        assert_eq!(escape("<a href=\"x\">R&C's</a>"), "&lt;a href=&quot;x&quot;&gt;R&amp;C&#39;s&lt;/a&gt;");
        assert_eq!((num(0), num(999), num(1000), num(1234567)), ("0".into(), "999".into(), "1,000".into(), "1,234,567".into()));
        assert_eq!(utc(0), "1970-01-01 00:00 UTC");
        assert_eq!(utc(1_793_909_675), "2026-11-05 20:14 UTC");
        assert_eq!(utc(951_782_400), "2000-02-29 00:00 UTC");
        let t = parse_titles("# x\nPPSA04263=GTA V\nPPSA00001=\nbad=Name\nPPSA01473 = Ratchet & Clank\n");
        assert_eq!(t.len(), 2);
        assert_eq!(t["PPSA01473"], "Ratchet & Clank");
        assert_eq!(parse_titles(TITLES)["PPSA21567"], "ASTRO BOT");
    }

    #[test]
    fn games_are_grouped_by_name_with_totals() {
        let mut page = Page { release: Some("v20".into()), ..Page::default() };
        page.titles.insert("PPSA00002".into(), "Zeta & <Co>".into());
        let all = [
            stats("PPSA04264", "01.011.001", [10, 7, 20, 5, 3, 4]),
            stats("PPSA00002", "1", [1, 0, 1, 0, 0, 0]),
            stats("PPSA04263", "01.011.001", [12, 12, 30, 30, 0, 2]),
            stats("PPSA21559", "01.000.000", [2, 0, 0, 0, 0, 0]),
            stats("PPSA99999", "01.000.000", [3, 1, 2, 1, 0, 0]),
            stats("TEST00001", "01.000.000", [100, 0, 0, 0, 0, 0]),
        ];
        let h = render(&page, &all, 1_793_909_675);
        // In name order: ASTRO BOT (21559 has no name of its own: its line of the regions table),
        // GTA V (both regions), the PPSA alone, then the escaped name.
        let order: Vec<usize> = ["<th colspan=\"10\" scope=\"colgroup\">ASTRO BOT", ">GTA V<", ">PPSA99999<", "Zeta &amp; &lt;Co&gt;"]
            .iter()
            .map(|n| h.find(n).unwrap_or_else(|| panic!("{n} missing")))
            .collect();
        assert!(order.windows(2).all(|w| w[0] < w[1]), "{order:?}");
        assert!(!h.contains("<Co>") && !h.contains("TEST00001"));
        assert!(h.find(">PPSA04263<").unwrap() < h.find(">PPSA04264<").unwrap());
        // Total, in the release, from players; pipelines likewise; copies apart.
        assert!(h.contains("<td class=\"l\">PPSA04264</td><td class=\"l\">01.011.001</td><td>10</td><td>7</td><td>3</td><td>20</td><td>5</td><td>15</td><td>3</td><td>4</td></tr>"));
        assert!(h.contains("<td class=\"l\">Total</td><td class=\"l\">4 games, 5 versions</td><td>28</td><td>20</td><td>8</td><td>53</td><td>36</td><td>17</td><td>3</td><td>6</td>"));
        assert!(h.contains("<th scope=\"col\">In v20</th>"));
        assert!(h.contains("Figures as of 2026-11-05 20:14 UTC, compared with the release v20."));
        assert!(h.contains("https://github.com/TheCruZ/KytyPS5-GTA/releases"));
        assert!(!h.contains("<script"));
        // No release: said so, and the note.
        let page = Page { note: Some("No release could be read: <main>.".into()), ..Page::default() };
        let h = render(&page, &[], 0);
        assert!(h.contains("Nothing recorded yet."));
        assert!(h.contains("with no release to compare with."));
        assert!(h.contains("No release could be read: &lt;main&gt;."));
    }
}
