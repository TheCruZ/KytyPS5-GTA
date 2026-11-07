//! The titles of one game in several regions (`regions.txt`, or the file of `--regions`).

use crate::validate;
use std::collections::HashMap;
use std::path::Path;

/// The built-in table.
pub const EMBEDDED: &str = include_str!("regions.txt");

/// One game: its name and its titles.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Game {
    pub name: String,
    pub titles: Vec<String>,
}

/// The games with more than one title. A title that is in none is a game of its own.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Regions {
    games: Vec<Game>,
    by_title: HashMap<String, usize>,
}

impl Regions {
    /// One game per line, `<name>: <title> <title> ...` (the last colon ends the name); `#` starts a
    /// comment. A title in two games, an invalid title or a game with a single title is an error.
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut regions = Self::default();
        for (number, raw) in text.lines().enumerate() {
            let line = raw.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            let at = |what: String| format!("line {}: {what}", number + 1);
            let Some((name, titles)) = line.rsplit_once(':') else {
                return Err(at("expected <name>: <titles>".into()));
            };
            let name = name.trim();
            if name.is_empty() {
                return Err(at("a game without a name".into()));
            }
            let titles: Vec<String> = titles.split_whitespace().map(str::to_string).collect();
            if titles.len() < 2 {
                return Err(at(format!("{name} has fewer than two titles")));
            }
            let index = regions.games.len();
            for title in &titles {
                if !validate::valid_title(title) {
                    return Err(at(format!("invalid title {title}")));
                }
                if regions.by_title.insert(title.clone(), index).is_some() {
                    return Err(at(format!("{title} is in two games")));
                }
            }
            regions.games.push(Game { name: name.to_string(), titles });
        }
        Ok(regions)
    }

    /// The built-in table (`regions.txt`).
    pub fn embedded() -> Self {
        Self::parse(EMBEDDED).unwrap_or_default()
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        Self::parse(&text)
    }

    pub fn games(&self) -> &[Game] {
        &self.games
    }

    /// Whether two titles are the same game (the same title, or two titles of one line).
    pub fn same_game(&self, a: &str, b: &str) -> bool {
        a == b || self.by_title.get(a).is_some_and(|g| self.by_title.get(b) == Some(g))
    }

    /// The name of the game of a title: its line, or the title itself.
    pub fn game_name<'a>(&'a self, title: &'a str) -> &'a str {
        self.by_title.get(title).map_or(title, |g| self.games[*g].name.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_table() {
        let r = Regions::embedded();
        let names: Vec<&str> = r.games().iter().map(|g| g.name.as_str()).collect();
        assert_eq!(names, ["GTA V", "ASTRO BOT", "Demon's Souls", "GTA: San Andreas - The Definitive Edition"]);
        assert!(r.same_game("PPSA04263", "PPSA04264"));
        assert!(r.same_game("PPSA21559", "PPSA21568"));
        assert!(r.same_game("PPSA03524", "PPSA03525"));
        assert!(r.same_game("PPSA26344", "PPSA26344"));
        assert!(!r.same_game("PPSA04263", "PPSA21564"));
        assert!(!r.same_game("PPSA26344", "PPSA26345"));
        assert_eq!(r.game_name("PPSA01342"), "Demon's Souls");
        assert_eq!(r.game_name("PPSA26344"), "PPSA26344");
    }

    #[test]
    fn bad_tables() {
        assert_eq!(Regions::parse("# nothing\n\n").unwrap(), Regions::default());
        for (text, why) in [
            ("A: PPSA00001", "line 1: A has fewer than two titles"),
            ("A PPSA00001 PPSA00002", "line 1: expected <name>: <titles>"),
            (": PPSA00001 PPSA00002", "line 1: a game without a name"),
            ("A: PPSA00001 ppsa00002", "line 1: invalid title ppsa00002"),
            ("A: PPSA00001 PPSA00002\nB: PPSA00003 PPSA00001", "line 2: PPSA00001 is in two games"),
        ] {
            assert_eq!(Regions::parse(text).unwrap_err(), why, "{text}");
        }
    }
}
