-- =====================================================================
-- NEXUS D2C Command Center: Supabase schema
-- Run this whole file once in Supabase -> SQL Editor -> New query -> Run.
-- It is safe to run again (everything uses IF NOT EXISTS / OR REPLACE).
-- =====================================================================

-- ---------- Helper: keep updated_at current ----------
create or replace function public.nexus_touch() returns trigger
language plpgsql set search_path = public as $$
begin
  new.updated_at := now();
  return new;
end $$;

-- ---------- products ----------
create table if not exists public.products (
  sku           text primary key,
  name          text not null check (length(trim(name)) > 0),
  category      text not null default 'General',
  price         numeric(12,2) not null check (price > 0),
  cost          numeric(12,2) not null default 0 check (cost >= 0),
  stock         integer not null default 0 check (stock >= 0),
  units_sold    integer not null default 0,
  status        text not null default 'active' check (status in ('active', 'paused')),
  ad_spend      numeric(14,2) not null default 0,
  base_rate     double precision,                    -- sample products: usual sales per day (builds history)
  website_rate  double precision not null default 1.0, -- expected website orders per day
  added_sim     double precision,                    -- new products: simulated day they were added
  created_at    timestamptz not null default clock_timestamp(),
  updated_at    timestamptz not null default now()
);
drop trigger if exists products_touch on public.products;
create trigger products_touch before update on public.products
  for each row execute function public.nexus_touch();

-- ---------- ads: one row per product per platform ----------
create table if not exists public.ads (
  sku            text not null references public.products(sku) on delete cascade,
  platform       text not null check (platform in ('Instagram','Facebook','YouTube','Google','Amazon','Flipkart','TikTok')),
  status         text not null default 'active' check (status in ('active', 'paused', 'stopped')),
  daily_budget   numeric(12,2) not null check (daily_budget > 0),
  started        text not null,
  seed_rate      double precision,   -- sample history: usual orders per day at seed_budget
  seed_budget    numeric(12,2),
  post_id        text,               -- the ad (post) currently running in this slot
  recent_reviews jsonb not null default '[]'::jsonb,  -- latest 20 written reviews
  created_at     timestamptz not null default now(),
  updated_at     timestamptz not null default now(),
  primary key (sku, platform)
);
drop trigger if exists ads_touch on public.ads;
create trigger ads_touch before update on public.ads
  for each row execute function public.nexus_touch();
create index if not exists ads_post_id_idx on public.ads (post_id);
create index if not exists ads_status_idx on public.ads (status);

-- ---------- posts: every ad posted through "Post a new ad" ----------
create table if not exists public.posts (
  id              text primary key,
  sku             text not null references public.products(sku) on delete cascade,
  product         text not null,
  platform        text not null,
  headline        text not null,
  "text"          text not null default '',
  cta             text not null default 'Shop now',
  format          text not null check (format in ('image', 'video', 'reel')),
  media_url       text not null default '',
  media_type      text not null default '',
  media_seconds   double precision,
  daily_budget    numeric(12,2) not null,
  duration_days   integer,
  posted_at       text not null,
  ends_sim        double precision,       -- simulated day the ad ends (null = runs until stopped)
  status          text not null default 'live' check (status in ('live', 'ended', 'replaced')),
  views           bigint not null default 0,
  clicks          bigint not null default 0,
  orders          integer not null default 0,
  sales           numeric(14,2) not null default 0,
  spend           numeric(14,2) not null default 0,
  reviews         integer not null default 0,
  rating_sum      double precision not null default 0,
  completed_views bigint not null default 0,
  watch_rate      double precision not null default 0,
  mode            text not null default 'demo',
  created_at      timestamptz not null default clock_timestamp()
);
create index if not exists posts_sku_idx on public.posts (sku);
create index if not exists posts_created_idx on public.posts (created_at desc);
create index if not exists posts_live_idx on public.posts (status, ends_sim);

-- ---------- orders: latest 300 orders synced from platforms ----------
create table if not exists public.orders (
  id         text primary key,
  "time"     text not null,             -- display time, e.g. "07 Oct, 02:15:09 PM"
  sku        text not null,             -- no foreign key: order history stays if a product is deleted
  product    text not null,
  platform   text not null,
  quantity   integer not null check (quantity > 0),
  amount     numeric(14,2) not null,
  created_at timestamptz not null default clock_timestamp()
);
create index if not exists orders_created_idx on public.orders (created_at desc);

-- ---------- monthly_metrics: totals per product, platform and month ----------
create table if not exists public.monthly_metrics (
  sku        text not null references public.products(sku) on delete cascade,
  platform   text not null,               -- an ad platform or 'Website'
  month      text not null check (month ~ '^[0-9]{4}-[0-9]{2}$'),
  units      bigint not null default 0,
  revenue    numeric(16,2) not null default 0,
  views      bigint not null default 0,
  clicks     bigint not null default 0,
  spend      numeric(16,2) not null default 0,
  reviews    bigint not null default 0,
  rating_sum double precision not null default 0,
  primary key (sku, platform, month)
);
create index if not exists monthly_month_idx on public.monthly_metrics (month);

-- ---------- connections: platform accounts, tokens encrypted by the backend ----------
create table if not exists public.connections (
  platform     text primary key check (platform in ('Instagram','Facebook','YouTube','Google','TikTok','Shopify','Amazon','Flipkart')),
  account_name text not null,
  public_ids   jsonb not null default '{}'::jsonb,  -- non-secret IDs (page ID, store address...)
  secrets_enc  jsonb not null default '{}'::jsonb,  -- Fernet ciphertext only; the key lives in NEXUS_SECRET_KEY
  masked       jsonb not null default '{}'::jsonb,  -- last 4 characters, for display
  mode         text not null default 'demo' check (mode in ('demo', 'live')),
  connected_at text,
  last_sync    text,
  last_test    text,
  updated_at   timestamptz not null default now()
);
drop trigger if exists connections_touch on public.connections;
create trigger connections_touch before update on public.connections
  for each row execute function public.nexus_touch();

-- ---------- actions: suggested changes on the Things To Do API ----------
create table if not exists public.actions (
  id         text primary key,
  type       text not null,
  target     text not null,
  change     text not null,
  impact     text not null,
  confidence double precision not null,
  status     text not null default 'pending',
  sort       integer not null default 0
);

-- ---------- app_state: small shared values (sync clock, one-time locks) ----------
create table if not exists public.app_state (
  key        text primary key,
  value      jsonb not null default '{}'::jsonb,
  updated_at timestamptz not null default now()
);

-- ---------- Views used by the backend (read with the caller's rights) ----------
create or replace view public.channel_totals with (security_invoker = true) as
  select sku, platform,
         sum(units)::bigint   as units,   sum(revenue) as revenue,
         sum(views)::bigint   as views,   sum(clicks)::bigint as clicks,
         sum(spend)           as spend,   sum(reviews)::bigint as reviews,
         sum(rating_sum)      as rating_sum
    from public.monthly_metrics
   group by sku, platform;

create or replace view public.sku_month_units with (security_invoker = true) as
  select sku, month, sum(units)::bigint as units
    from public.monthly_metrics
   group by sku, month;

-- =====================================================================
-- Functions (each runs in one transaction, so several servers can share the data safely)
-- =====================================================================

-- Sync clock: created once, shared by every server copy.
create or replace function public.nexus_clock() returns jsonb
language plpgsql set search_path = public as $$
declare v jsonb;
begin
  insert into app_state(key, value)
  values ('clock', jsonb_build_object('start', extract(epoch from now()),
                                      'last_sync', extract(epoch from now()),
                                      'label', ''))
  on conflict (key) do nothing;
  select value into v from app_state where key = 'clock';
  return v;
end $$;

-- Claim the next order sync: returns the previous sync time and moves it to now (row is locked).
create or replace function public.nexus_claim_sync(p_label text) returns jsonb
language plpgsql set search_path = public as $$
declare
  v    jsonb;
  prev double precision;
  nowt double precision := extract(epoch from clock_timestamp());
begin
  perform nexus_clock();
  select value into v from app_state where key = 'clock' for update;
  prev := (v->>'last_sync')::double precision;
  update app_state
     set value = v || jsonb_build_object('last_sync', nowt, 'label', p_label), updated_at = now()
   where key = 'clock';
  return jsonb_build_object('start', (v->>'start')::double precision, 'prev', prev, 'now', nowt);
end $$;

-- One-time lock (used so sample data is created only once). True the first time only.
create or replace function public.nexus_try_lock(p_key text) returns boolean
language plpgsql set search_path = public as $$
begin
  insert into app_state(key, value) values (p_key, jsonb_build_object('at', now()));
  return true;
exception when unique_violation then
  return false;
end $$;

-- Add stock without losing a sale that happens at the same moment.
create or replace function public.nexus_add_stock(p_sku text, p_qty integer) returns jsonb
language plpgsql set search_path = public as $$
declare prod products;
begin
  update products set stock = stock + p_qty where sku = p_sku returning * into prod;
  if not found then
    return null;
  end if;
  return to_jsonb(prod);
end $$;

-- Apply one order sync: stock, month totals, per-ad numbers, new orders, reviews. All or nothing.
create or replace function public.nexus_apply_sync(p jsonb) returns void
language plpgsql set search_path = public as $$
declare r jsonb;
begin
  -- 1. take sold units out of stock
  for r in select value from jsonb_array_elements(coalesce(p->'stock', '[]'::jsonb)) loop
    update products
       set stock = greatest(stock - (r->>'qty')::integer, 0),
           units_sold = units_sold + (r->>'qty')::integer
     where sku = r->>'sku';
  end loop;

  -- 2. add to the month totals (one row per product + platform + month in each payload)
  insert into monthly_metrics as m (sku, platform, month, units, revenue, views, clicks, spend, reviews, rating_sum)
  select x.sku, x.platform, x.month, x.units, x.revenue, x.views, x.clicks, x.spend, x.reviews, x.rating_sum
    from jsonb_to_recordset(coalesce(p->'monthly', '[]'::jsonb))
         as x(sku text, platform text, month text, units bigint, revenue numeric, views bigint,
              clicks bigint, spend numeric, reviews bigint, rating_sum double precision)
   where exists (select 1 from products pr where pr.sku = x.sku)
  on conflict (sku, platform, month) do update set
    units      = m.units      + excluded.units,
    revenue    = m.revenue    + excluded.revenue,
    views      = m.views      + excluded.views,
    clicks     = m.clicks     + excluded.clicks,
    spend      = m.spend      + excluded.spend,
    reviews    = m.reviews    + excluded.reviews,
    rating_sum = m.rating_sum + excluded.rating_sum;

  -- 3. numbers for each posted ad
  update posts as po set
    views           = po.views + x.views,
    clicks          = po.clicks + x.clicks,
    orders          = po.orders + x.orders,
    sales           = po.sales + x.sales,
    spend           = po.spend + x.spend,
    reviews         = po.reviews + x.reviews,
    rating_sum      = po.rating_sum + x.rating_sum,
    completed_views = po.completed_views + x.completed_views
  from jsonb_to_recordset(coalesce(p->'posts', '[]'::jsonb))
       as x(id text, views bigint, clicks bigint, orders integer, sales numeric, spend numeric,
            reviews integer, rating_sum double precision, completed_views bigint)
  where po.id = x.id;

  -- 4. the new orders
  insert into orders (id, "time", sku, product, platform, quantity, amount)
  select x.id, x."time", x.sku, x.product, x.platform, x.quantity, x.amount
    from jsonb_to_recordset(coalesce(p->'orders', '[]'::jsonb))
         as x(id text, "time" text, sku text, product text, platform text, quantity integer, amount numeric)
  on conflict (id) do nothing;

  -- 5. newest written reviews on each ad, keeping the latest 20
  for r in select value from jsonb_array_elements(coalesce(p->'reviews', '[]'::jsonb)) loop
    update ads set recent_reviews = (
      select coalesce(jsonb_agg(last20.e order by last20.ord), '[]'::jsonb)
        from (select t.e, t.ord
                from jsonb_array_elements(ads.recent_reviews || coalesce(r->'items', '[]'::jsonb))
                     with ordinality as t(e, ord)
               order by t.ord desc
               limit 20) as last20)
     where sku = r->>'sku' and platform = r->>'platform';
  end loop;

  -- 6. keep only the latest 300 orders
  delete from orders
   where id in (select id from orders order by created_at desc offset 300);
end $$;

-- =====================================================================
-- Security: only the backend (service_role key) may read or change anything.
-- Row Level Security is on with no policies, so the public anon key sees nothing.
-- =====================================================================
alter table public.products        enable row level security;
alter table public.ads             enable row level security;
alter table public.posts           enable row level security;
alter table public.orders          enable row level security;
alter table public.monthly_metrics enable row level security;
alter table public.connections     enable row level security;
alter table public.actions         enable row level security;
alter table public.app_state       enable row level security;

revoke all on public.products, public.ads, public.posts, public.orders, public.monthly_metrics,
              public.connections, public.actions, public.app_state,
              public.channel_totals, public.sku_month_units
  from anon, authenticated;

revoke execute on function public.nexus_clock(), public.nexus_claim_sync(text), public.nexus_try_lock(text),
                           public.nexus_add_stock(text, integer), public.nexus_apply_sync(jsonb)
  from public, anon, authenticated;
grant execute on function public.nexus_clock(), public.nexus_claim_sync(text), public.nexus_try_lock(text),
                          public.nexus_add_stock(text, integer), public.nexus_apply_sync(jsonb)
  to service_role;
grant select, insert, update, delete on public.products, public.ads, public.posts, public.orders,
      public.monthly_metrics, public.connections, public.actions, public.app_state to service_role;
grant select on public.channel_totals, public.sku_month_units to service_role;

-- =====================================================================
-- Storage: public bucket for ad pictures and videos
-- file_size_limit 50 MB = the Supabase Free plan maximum. On Pro you can raise it
-- (e.g. 104857600 for 100 MB) and set MAX_VIDEO_MB=100 in Railway.
-- =====================================================================
insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values ('ad_creatives', 'ad_creatives', true, 52428800,
        array['image/jpeg', 'image/png', 'image/webp', 'video/mp4', 'video/webm', 'video/quicktime'])
on conflict (id) do update
  set public = excluded.public,
      file_size_limit = excluded.file_size_limit,
      allowed_mime_types = excluded.allowed_mime_types;
-- Public bucket = anyone can VIEW files by their URL (needed so ads display).
-- Uploads go through the backend with the service_role key, so no upload policy is needed.
