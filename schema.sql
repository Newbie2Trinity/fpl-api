create table if not exists app_state (
  id int primary key default 1,
  squad jsonb,
  captain_id int,
  bank numeric default 0,
  free_transfers int default 1,
  gameweek int,
  chips_used jsonb default '[]'::jsonb,
  updated_at timestamptz default now()
);
insert into app_state (id) values (1) on conflict (id) do nothing;
alter table app_state enable row level security;

-- If you already deployed this table before the chip advisor feature was
-- added, run this instead of the create table above:
-- alter table app_state add column if not exists chips_used jsonb default '[]'::jsonb;

-- Last season's game-time % per player, used as the xP model's preseason
-- "will they actually start" prior (see refresh_history_cache() in _lib.py
-- and api/refresh_history_cache.py). Populated by that endpoint, not by
-- hand -- this table starts empty and is fine to be empty (the model just
-- falls back to the ownership-based prior until it's populated).
create table if not exists player_history_cache (
  player_id int primary key,
  last_season_minutes int,
  last_season_game_time_pct numeric,
  updated_at timestamptz default now()
);
alter table player_history_cache enable row level security;
