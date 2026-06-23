import asyncio
import base64
from contextlib import asynccontextmanager
import datetime
import json
import os
import uuid
from fastapi import *
from fastapi.sse import EventSourceResponse
import psycopg
from fastapi.middleware.cors import CORSMiddleware
import aio_pika
from pathlib import Path

rabbit_connection = None
consumer_task = None

results = {}

async def rabbit_consumer(
    connection
):
    channel = await connection.channel()
    queue = await channel.declare_queue(
        "results",
        durable=True
    )

    async with queue.iterator() as iterator:
        async for message in iterator:
            async with message.process():
                if message.correlation_id in results: 
                    results[message.correlation_id] = message.body

@asynccontextmanager
async def lifespan(
    app: FastAPI
):
    with psycopg.connect("dbname=orchestratordb user=user password=123 host=db-orchestrator port=5431") as conn:
        with conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS experiments_runs (
                id SERIAL PRIMARY KEY,
                cell_id INT,
                seq_no INT,
                method_name TEXT,
                config TEXT,
                status TEXT,
                started_at TIMESTAMP,
                finished_at TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS experiments_runs_results (
                id SERIAL PRIMARY KEY,
                run_id INT,
                text_json JSONB,
                metrics_json JSONB,
                artifacts_refs_json JSONB
            )
            """)
            conn.commit()

    global rabbit_connection
    rabbit_connection = await aio_pika.connect_robust("amqp://guest:guest@rabbitmq:5672/")

    global consumer_task
    consumer_task = asyncio.create_task(rabbit_consumer(rabbit_connection))

    yield

    consumer_task.cancel()
    await consumer_task

    await rabbit_connection.close()

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/results/{task_id}")
async def results_event(
    task_id: str
):
    async def generator():
        while True:
            result = results.get(task_id)

            if result is not None:
                data = json.loads(result)

                if data:
                    figures_saved = {} 
                    if "fig" in data:
                        output_file = Path(f"./stored/{task_id}.json")
                        output_file.parent.mkdir(exist_ok=True, parents=True)

                        with open(output_file, "w", encoding="utf-8") as f:
                            f.write(data["fig"])
                        del data["fig"]
                        figures_saved['fig'] = f'{task_id}.json'
                    
                    with psycopg.connect("dbname=orchestratordb user=user password=123 host=db-orchestrator port=5431") as conn:
                        with conn.cursor() as cur:
                            cur.execute(f"""
                                UPDATE experiments_runs
                                SET status = 'finished',
                                    finished_at = '{datetime.datetime.now(datetime.timezone.utc)}'
                                WHERE id = {task_id}
                            """)
                            
                            cur.execute("""
                                INSERT INTO experiments_runs_results (
                                    run_id,
                                    text_json,
                                    metrics_json,
                                    artifacts_refs_json
                                )
                                VALUES (%s,%s,%s,%s)
                            """, (task_id, json.dumps(data) if len(data)!=0 else None, None, json.dumps(figures_saved) if figures_saved else None))
                            
                        conn.commit()

                    yield f"data: {result.decode("utf-8")}\n\n"
                    del results[task_id]
                    break
                    
                else:
                    with psycopg.connect("dbname=orchestratordb user=user password=123 host=db-orchestrator port=5431") as conn:
                        with conn.cursor() as cur:
                            cur.execute(f"""
                                DELETE 
                                FROM experiments_runs
                                WHERE id = {task_id}
                            """)

                        conn.commit()

                    yield "data: {}\n\n"
                    del results[task_id]
                    break

            await asyncio.sleep(0.5)

    return EventSourceResponse(generator())

@app.get("/results")
async def results_test():
    content = {
        "data": list(results.keys())
    }
    return Response(content=json.dumps(content), media_type="application/json")

@app.post("/method")
async def method(
    request: Request, 
    data: dict
):
    channel = await rabbit_connection.channel()

    queue = await channel.declare_queue(
        "main_queue",
        durable=True
    )

    with psycopg.connect("dbname=orchestratordb user=user password=123 host=db-orchestrator port=5431") as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO experiments_runs (
                    cell_id,
                    seq_no,
                    method_name,
                    config,
                    status,
                    started_at,
                    finished_at
                )
                VALUES (%s,%s,%s,%s,'pending',%s, NULL)
                RETURNING id;
            """, (
                data['cell_id'],
                data['seq_no'],
                data['method'],
                data['config'],
                datetime.datetime.now(datetime.timezone.utc)
            ))
            
            inserted_id = cur.fetchone()[0]
            conn.commit()

    results[str(inserted_id)] = None
    
    await channel.default_exchange.publish(
        aio_pika.Message(
            body=json.dumps(data).encode(),
            correlation_id=inserted_id,
            reply_to="results"
        ),
        routing_key=queue.name
    )

    content = {
        "cid": inserted_id
    }
    
    return Response(content=json.dumps(content), media_type="application/json")

@app.get("/composition/{task_id}/data")
async def composition_data(
    task_id: str
):
    return Response(
        content=results.pop(task_id),
        media_type="application/octet-stream"
    )

@app.get("/composition/{task_id}")
async def composition_result(
    task_id: str
):
    async def generator():
        while True:
            result = results.get(task_id)

            if result is not None:
                result = json.loads(result)

                binary = base64.b64decode(
                    result.pop("result")
                )

                results[task_id] = binary

                yield f"data: {json.dumps(result)}\n\n"

                break
            
            await asyncio.sleep(0.5)
    
    return EventSourceResponse(generator())

@app.post("/composition")
async def composition(
    request: Request, 
    data: dict
):
    channel = await rabbit_connection.channel()

    queue = await channel.declare_queue(
        "main_queue",
        durable=True
    )

    idx = uuid.uuid4().hex
    results[idx] = None
    
    await channel.default_exchange.publish(
        aio_pika.Message(
            body=json.dumps(data).encode(),
            correlation_id=idx,
            reply_to="results"
        ),
        routing_key=queue.name
    )

    content = {
        "cid": idx
    }
    
    return Response(content=json.dumps(content), media_type="application/json")

@app.post("/history")
async def get_history(
    request: Request, 
    data: dict
):
    with psycopg.connect("dbname=orchestratordb user=user password=123 host=db-orchestrator port=5431", row_factory=psycopg.rows.dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, method_name, config
                FROM experiments_runs
                WHERE cell_id=%s AND status='finished'
                ORDER BY finished_at DESC
            """, (data['cell_id'],))
            
            history = cur.fetchall()
            
    content = {
        "history": history
    }
    
    return Response(content=json.dumps(content), media_type="application/json")

@app.delete("/history/{id}/delete")
async def delete_history(
    request: Request, 
    id: int
):
    with psycopg.connect("dbname=orchestratordb user=user password=123 host=db-orchestrator port=5431", row_factory=psycopg.rows.dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                DELETE
                FROM experiments_runs
                WHERE id=%s
            """, (id,))
            
            cur.execute("""
                DELETE
                FROM experiments_runs_results
                WHERE run_id=%s
                RETURNING *
            """, (id,))

            operation = cur.fetchone()
            if operation['artifacts_refs_json']['fig'] and os.path.exists(f"./stored/{operation['artifacts_refs_json']['fig']}"):
                os.remove(f"./stored/{operation['artifacts_refs_json']['fig']}")

        conn.commit()

    return Response(status_code=status.HTTP_204_NO_CONTENT)

@app.post("/history/{id}")
async def from_history(
    request: Request, 
    id: int
):
    with psycopg.connect("dbname=orchestratordb user=user password=123 host=db-orchestrator port=5431", row_factory=psycopg.rows.dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT method_name, config
                FROM experiments_runs
                WHERE id=%s
            """, (id,))
            
            method = cur.fetchone()
            
            cur.execute("""
                SELECT * 
                FROM experiments_runs_results
                WHERE run_id=%s
            """, (id,))

            operation = cur.fetchone()
    
    content = {
        'method': method
    }
    
    artifacts = operation.get('artifacts_refs_json')
    if artifacts and artifacts.get('fig'):
        with open(f"./stored/{operation['artifacts_refs_json']['fig']}", "r", encoding="utf-8") as f:
            content["fig"] = f.read()

    return Response(content=json.dumps(content | operation['text_json']), media_type="application/json")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("orchestrator:app", port=81, log_level="info", reload=True)