-- DuckDB schema for MANTIS miner

CREATE TABLE IF NOT EXISTS prices (
    asset      VARCHAR NOT NULL,
    ts         TIMESTAMP NOT NULL,
    open       DOUBLE,
    high       DOUBLE,
    low        DOUBLE,
    close      DOUBLE,
    volume     DOUBLE,
    PRIMARY KEY (asset, ts)
);

CREATE TABLE IF NOT EXISTS funding_rates (
    asset      VARCHAR NOT NULL,
    ts         TIMESTAMP NOT NULL,
    rate       DOUBLE,
    oi         DOUBLE,
    long_short_ratio DOUBLE,
    PRIMARY KEY (asset, ts)
);

CREATE TABLE IF NOT EXISTS liquidations (
    asset      VARCHAR NOT NULL,
    ts         TIMESTAMP NOT NULL,
    side       VARCHAR,       -- 'long' or 'short'
    amount_usd DOUBLE,
    PRIMARY KEY (asset, ts, side)
);

CREATE TABLE IF NOT EXISTS onchain_flows (
    asset            VARCHAR NOT NULL,
    ts               TIMESTAMP NOT NULL,
    exchange_inflow  DOUBLE,
    exchange_outflow DOUBLE,
    whale_tx_count   INTEGER,
    reserve_change   DOUBLE,
    PRIMARY KEY (asset, ts)
);

CREATE TABLE IF NOT EXISTS sentiment (
    asset          VARCHAR NOT NULL,
    ts             TIMESTAMP NOT NULL,
    social_volume  DOUBLE,
    engagement     DOUBLE,
    galaxy_score   DOUBLE,
    social_dominance DOUBLE,
    PRIMARY KEY (asset, ts)
);

CREATE TABLE IF NOT EXISTS fear_greed (
    ts             TIMESTAMP PRIMARY KEY,
    value          INTEGER,
    classification VARCHAR
);

CREATE TABLE IF NOT EXISTS features (
    ts          TIMESTAMP NOT NULL,
    challenge   VARCHAR NOT NULL,
    asset       VARCHAR NOT NULL,
    features    BLOB,         -- packed numpy array
    PRIMARY KEY (ts, challenge, asset)
);

CREATE TABLE IF NOT EXISTS predictions (
    ts          TIMESTAMP NOT NULL,
    challenge   VARCHAR NOT NULL,
    embedding   VARCHAR,      -- JSON string
    PRIMARY KEY (ts, challenge)
);

CREATE TABLE IF NOT EXISTS labels (
    ts          TIMESTAMP NOT NULL,
    challenge   VARCHAR NOT NULL,
    asset       VARCHAR NOT NULL,
    label       DOUBLE,
    PRIMARY KEY (ts, challenge, asset)
);
