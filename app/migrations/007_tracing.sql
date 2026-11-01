-- LR3 metadata only: no message bodies, arguments, capabilities or destinations.
CREATE TABLE labbi_traces (
    id UUID PRIMARY KEY,
    request_id UUID NOT NULL,
    user_id BIGINT NOT NULL REFERENCES labbi_users(user_id) ON DELETE RESTRICT,
    started_at TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ,
    status TEXT NOT NULL CHECK (status IN ('running','ok','failed','cancelled','interrupted')),
    CHECK ((status = 'running') = (finished_at IS NULL)),
    UNIQUE (id, request_id)
);
CREATE INDEX labbi_traces_owner_idx ON labbi_traces(user_id, started_at DESC, id);
CREATE TABLE labbi_trace_spans (
    id UUID PRIMARY KEY,
    trace_id UUID NOT NULL,
    request_id UUID NOT NULL,
    parent_id UUID,
    data JSONB NOT NULL CHECK (jsonb_typeof(data) = 'object'),
    UNIQUE (id, trace_id, request_id),
    FOREIGN KEY (trace_id, request_id) REFERENCES labbi_traces(id, request_id),
    FOREIGN KEY (parent_id, trace_id, request_id)
        REFERENCES labbi_trace_spans(id, trace_id, request_id)
);
