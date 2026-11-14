DROP TABLE IF EXISTS readings;
DROP TABLE IF EXISTS parameters;
DROP TABLE IF EXISTS wells;

CREATE TABLE wells (
    well_id   TEXT PRIMARY KEY,
    well_type TEXT NOT NULL
);

CREATE TABLE parameters (
    parameter TEXT PRIMARY KEY,
    unit      TEXT NOT NULL,
    well_type TEXT NOT NULL
);

CREATE TABLE readings (
    reading_id INTEGER PRIMARY KEY,
    well_id    TEXT NOT NULL REFERENCES wells (well_id),
    parameter  TEXT NOT NULL REFERENCES parameters (parameter),
    timestamp  TEXT NOT NULL,
    value      REAL,
    n_obs      INTEGER,
    flag       TEXT,
    split      TEXT NOT NULL
);

CREATE INDEX idx_readings_well_param_ts ON readings (well_id, parameter, timestamp);
CREATE INDEX idx_readings_ts ON readings (timestamp);
