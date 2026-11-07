//! Rate limits. Everything is keyed by salted hashes, never by clear IPs.

use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct LimitConfig {
    /// Requests per IP in each window.
    pub ip_requests: u32,
    pub ip_window_secs: u64,
    /// New records per client and per IP per UTC day.
    pub client_daily: u64,
    pub ip_daily: u64,
}

impl Default for LimitConfig {
    fn default() -> Self {
        Self { ip_requests: 60, ip_window_secs: 600, client_daily: 20_000, ip_daily: 100_000 }
    }
}

const DAY: u64 = 86_400;

#[derive(Default)]
pub struct Limits {
    cfg: LimitConfig,
    ip_req: HashMap<u64, (u64, u32)>,
    client_daily: HashMap<u64, (u64, u64)>,
    ip_daily: HashMap<u64, (u64, u64)>,
}

impl Limits {
    pub fn new(cfg: LimitConfig) -> Self {
        Self { cfg, ..Self::default() }
    }

    /// Counts a request of `ip`; `Err(seconds)` is the `Retry-After` when it is over the limit.
    pub fn check_ip(&mut self, ip: u64, now: u64) -> Result<(), u64> {
        let window = self.cfg.ip_window_secs;
        let entry = self.ip_req.entry(ip).or_insert((now, 0));
        if now >= entry.0 + window {
            *entry = (now, 0);
        }
        if entry.1 >= self.cfg.ip_requests {
            return Err((entry.0 + window).saturating_sub(now).max(1));
        }
        entry.1 += 1;
        Ok(())
    }

    fn used(map: &HashMap<u64, (u64, u64)>, key: u64, now: u64) -> u64 {
        match map.get(&key) {
            Some((day, n)) if *day == now / DAY => *n,
            _ => 0,
        }
    }

    /// How many new records this client from this IP may still add today.
    pub fn remaining_new(&self, client: u64, ip: u64, now: u64) -> u64 {
        let by_client = self.cfg.client_daily.saturating_sub(Self::used(&self.client_daily, client, now));
        let by_ip = self.cfg.ip_daily.saturating_sub(Self::used(&self.ip_daily, ip, now));
        by_client.min(by_ip)
    }

    pub fn add_new(&mut self, client: u64, ip: u64, n: u64, now: u64) {
        let day = now / DAY;
        for (map, key) in [(&mut self.client_daily, client), (&mut self.ip_daily, ip)] {
            let entry = map.entry(key).or_insert((day, 0));
            if entry.0 != day {
                *entry = (day, 0);
            }
            entry.1 += n;
        }
    }

    /// Seconds to the next UTC day, when the daily counters restart.
    pub fn secs_to_next_day(now: u64) -> u64 {
        DAY - now % DAY
    }

    /// Drops the expired entries.
    pub fn gc(&mut self, now: u64) {
        let window = self.cfg.ip_window_secs;
        self.ip_req.retain(|_, (start, _)| now < *start + window);
        let day = now / DAY;
        self.client_daily.retain(|_, (d, _)| *d == day);
        self.ip_daily.retain(|_, (d, _)| *d == day);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ip_window() {
        let mut l = Limits::new(LimitConfig { ip_requests: 3, ip_window_secs: 100, ..Default::default() });
        for _ in 0..3 {
            assert!(l.check_ip(1, 1000).is_ok());
        }
        assert_eq!(l.check_ip(1, 1040), Err(60));
        assert!(l.check_ip(2, 1040).is_ok(), "another IP is independent");
        assert!(l.check_ip(1, 1100).is_ok(), "the window restarted");
        l.gc(5000);
        assert!(l.ip_req.is_empty());
    }

    #[test]
    fn daily_records() {
        let mut l = Limits::new(LimitConfig { client_daily: 10, ip_daily: 15, ..Default::default() });
        let t = 3 * DAY + 5;
        assert_eq!(l.remaining_new(1, 7, t), 10);
        l.add_new(1, 7, 8, t);
        assert_eq!(l.remaining_new(1, 7, t), 2);
        assert_eq!(l.remaining_new(2, 7, t), 7, "the IP limit is shared by clients");
        assert_eq!(l.remaining_new(1, 7, t + DAY), 10, "next day");
        assert_eq!(Limits::secs_to_next_day(t), DAY - 5);
    }
}
