"""Short transactions; owner-scoped reads and closed DTO metadata only."""

from uuid import UUID

from app.routing.contracts import TraceSpan


class TraceRepository:
    def __init__(self, pool):
        self.pool = pool

    async def start(self, owner, trace_id, request_id, started_at):
        if type(owner) is not int or not 0 < owner <= 2**63 - 1:
            raise ValueError("Invalid owner")
        async with self.pool.acquire() as conn, conn.transaction():
            await conn.execute(
                "INSERT INTO labbi_users(user_id) VALUES($1) ON CONFLICT DO NOTHING", owner
            )
            await conn.execute(
                "INSERT INTO labbi_traces(id,request_id,user_id,started_at,status) "
                "VALUES($1,$2,$3,$4,'running')",
                trace_id,
                request_id,
                owner,
                started_at,
            )

    async def span(self, span: TraceSpan):
        span = TraceSpan.model_validate(span)
        await self.pool.execute(
            "INSERT INTO labbi_trace_spans(id,trace_id,request_id,parent_id,data) "
            "VALUES($1,$2,$3,$4,$5::jsonb) ON CONFLICT(id) DO UPDATE SET data=EXCLUDED.data "
            "WHERE labbi_trace_spans.trace_id=EXCLUDED.trace_id "
            "AND labbi_trace_spans.request_id=EXCLUDED.request_id "
            "AND labbi_trace_spans.parent_id IS NOT DISTINCT FROM EXCLUDED.parent_id",
            span.span_id,
            span.trace_id,
            span.request_id,
            span.parent_span_id,
            span.model_dump_json(),
        )

    async def finish(self, trace_id, status, finished_at):
        await self.pool.execute(
            "UPDATE labbi_traces SET status=$2,finished_at=$3 WHERE id=$1 AND status='running'",
            trace_id,
            status,
            finished_at,
        )

    async def read(self, owner, identifier=None):
        # A crashed request is visible as interrupted after its maximum execution window.
        async with self.pool.acquire() as conn, conn.transaction():
            await conn.execute(
                "UPDATE labbi_traces SET status='interrupted',finished_at=now() "
                "WHERE user_id=$1 AND status='running' "
                "AND started_at < now() - INTERVAL '10 minutes'",
                owner,
            )
            row = await conn.fetchrow(
                "SELECT * FROM labbi_traces WHERE user_id=$1 "
                "AND ($2::uuid IS NULL OR id=$2) ORDER BY started_at DESC,id DESC LIMIT 1",
                owner,
                UUID(str(identifier)) if identifier is not None else None,
            )
            if row is None:
                return None
            rows = await conn.fetch(
                "SELECT data FROM labbi_trace_spans WHERE trace_id=$1 ORDER BY "
                "(data->>'started_at')::timestamptz,id",
                row["id"],
            )
        spans = [TraceSpan.model_validate_json(r["data"]) for r in rows]
        return dict(row), spans

    async def display(self, owner, argument):
        try:
            identifier = None if argument in (None, "", "last") else UUID(argument)
        except (ValueError, TypeError, AttributeError):
            return "Используй /trace last или /trace <UUID>."
        found = await self.read(owner, identifier)
        if found is None:
            return "Трасса не найдена. Доступны только твои запросы."
        row, spans = found
        lines = [f"Трасса: {row['id']}", f"Статус: {row['status']}"]
        for span in spans[:64]:
            label = span.tool_name or span.resource_uri or span.target or ""
            usage = (
                f"; токены {span.input_tokens}/{span.output_tokens}"
                if span.input_tokens is not None or span.output_tokens is not None
                else ""
            )
            lines.append(
                f"• {span.component}/{span.operation} {label}: {span.status}, "
                f"{span.duration_ms if span.duration_ms is not None else '…'} мс{usage}"
            )
        if any(s.operation.startswith("model_") and s.input_tokens is None for s in spans):
            lines.append("Токены без usage провайдера неизвестны; оценки не подставляются.")
        if row["status"] in {"running", "interrupted"}:
            lines.append("Трасса не завершена; результат доставки может быть неизвестен.")
        return "\n".join(lines)
