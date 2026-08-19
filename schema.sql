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
