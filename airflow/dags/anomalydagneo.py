import os
import pendulum
from datetime import timedelta
import logging

from airflow.models.dag import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.influxdb.hooks.influxdb import InfluxDBHook

TZ = "Europe/Madrid"

def rfc3339_utc(dt):
    # Tiempos en UTC con Z para Flux
    return pendulum.instance(dt).in_timezone('UTC').to_iso8601_string().replace('+00:00', 'Z')

def parse_topic_to_tags(topic: str):
    segs = (topic or "").split("/")
    segs += [None] * max(0, 6 - len(segs))
    return segs[0], segs[1], segs[2], segs[3]  # company, site, line, device


def get_sensor_query(sensor_config: dict, start_time: str, stop_time: str) -> str:
    company, site, line, device = parse_topic_to_tags(sensor_config.get('topic', ''))
    return f'''
from(bucket: "iotlab")
  |> range(start: time(v: "{start_time}"), stop: time(v: "{stop_time}"))
  |> filter(fn: (r) => r["_measurement"] == "{sensor_config['measurement']}")
  |> filter(fn: (r) => r["_field"] == "{sensor_config.get('field','value')}")
  |> filter(fn: (r) => r["company"] == "{company}")
  |> filter(fn: (r) => r["site"] == "{site}")
  |> filter(fn: (r) => r["line"] == "{line}")
  |> filter(fn: (r) => r["device"] == "{device}")
  |> aggregateWindow(every: 1m, fn: mean, createEmpty: false)
  |> yield(name: "sensor_data")
'''

class SimplePercentileAD:
    def __init__(self, percentiles=(5, 95)):
        self.percentiles = percentiles
        self.thresholds = None

    def fit(self, X, y=None):
        import numpy as np
        self.thresholds = np.percentile(X, self.percentiles)
        return self

    def predict(self, X):
        if self.thresholds is None:
            raise ValueError("Model must be fitted before prediction")
        return (X < self.thresholds[0]) | (X > self.thresholds[1])

def load_asset_config_from_neo4j(**kwargs):
    """
    Carga config desde Neo4j sin asumir label en el extremo de MEASURE_OF.
    1) MATCH (m:Measure)-[:MEASURE_OF]-()
    2) Fallback: MATCH (m:Measure)
    """
    from neo4j import GraphDatabase

    bolt = os.getenv('NEO4J_BOLT_URL', 'bolt://localhost:7687')
    user = os.getenv('NEO4J_USER', 'neo4j')
    pwd  = os.getenv('NEO4J_PASSWORD', 'admin')
    database = os.getenv('NEO4J_DATABASE', None)

    q_by_rel = """
    MATCH (m:Measure)-[:MEASURE_OF]-()
    RETURN m.alias AS alias,
           m.topic AS topic,
           m.measurement AS measurement,
           coalesce(m.field,'value') AS field,
           coalesce(m.window_minutes, 5) AS window_minutes,
           coalesce(m.p_low, 5) AS p_low,
           coalesce(m.p_high, 95) AS p_high
    """
    q_minimal = """
    MATCH (m:Measure)
    RETURN m.alias AS alias,
           m.topic AS topic,
           m.measurement AS measurement,
           coalesce(m.field,'value') AS field,
           coalesce(m.window_minutes, 5) AS window_minutes,
           coalesce(m.p_low, 5) AS p_low,
           coalesce(m.p_high, 95) AS p_high
    """

    cfg = {}
    driver = GraphDatabase.driver(bolt, auth=(user, pwd))
    try:
        sess_kwargs = {}
        if database:
            sess_kwargs["database"] = database
        with driver.session(**sess_kwargs) as s:
            rows = list(s.run(q_by_rel))
            if not rows:
                logging.warning("[Neo4j] No rows with undirected MEASURE_OF; falling back to minimal match.")
                rows = list(s.run(q_minimal))
            for row in rows:
                alias = row.get('alias')
                if not alias:
                    continue
                cfg[alias] = {
                    'topic': row.get('topic'),
                    'measurement': row.get('measurement'),
                    'field': row.get('field'),
                    'window_minutes': int(row.get('window_minutes')),
                    'threshold_percentiles': (float(row.get('p_low')), float(row.get('p_high'))),
                }
    finally:
        driver.close()

    logging.info(f"[Neo4j] Loaded measures: {list(cfg.keys())}")
    if not cfg:
        logging.error("[Neo4j] No measures found. Verify labels/properties.")
    return cfg

def validate_sources(**kwargs):
    ti = kwargs['ti']
    dynamic_config = ti.xcom_pull(task_ids='load_asset_config') or {}
    hook = InfluxDBHook(conn_id="influxdb")
    now_local = pendulum.now(TZ)

    for alias, scfg in dynamic_config.items():
        test_start = now_local - timedelta(minutes=scfg.get('window_minutes', 5))
        test_stop = now_local
        q = get_sensor_query(scfg, rfc3339_utc(test_start), rfc3339_utc(test_stop))
        try:
            df = hook.query_to_df(q)
            n = 0 if df is None else int(df.shape[0])
            logging.info(f"[VALIDATE] Alias '{alias}': puntos en ventana = {n}")
            if n == 0:
                logging.warning(f"[VALIDATE] No data for alias '{alias}' in last {scfg.get('window_minutes',5)} min.")
        except Exception as e:
            logging.error(f"[VALIDATE] Error querying alias '{alias}': {e}")


def detect_anomalies_for_sensor(sensor_name, **kwargs):
    import numpy as np
    from requests.exceptions import SSLError

    ti = kwargs['ti']
    dynamic_config = ti.xcom_pull(task_ids='load_asset_config') or {}
    if sensor_name not in dynamic_config:
        logging.warning(f"No config in Neo4j for alias '{sensor_name}'.")
        return {'sensor': sensor_name, 'anomalies_count': 0, 'total_points': 0, 'anomaly_rate': 0.0}

    sensor_config = dynamic_config[sensor_name]
    hook = InfluxDBHook(conn_id="influxdb")

    now_local = pendulum.now(TZ)
    train_start = now_local - timedelta(hours=1)
    train_stop  = now_local - timedelta(minutes=sensor_config['window_minutes'])
    test_start  = now_local - timedelta(minutes=sensor_config['window_minutes'])
    test_stop   = now_local

    train_query = get_sensor_query(sensor_config, rfc3339_utc(train_start), rfc3339_utc(train_stop))
    test_query  = get_sensor_query(sensor_config, rfc3339_utc(test_start),  rfc3339_utc(test_stop))

    try:
        df_train = hook.query_to_df(train_query)
        if df_train is None or df_train.empty or df_train["_value"].dropna().shape[0] < 10:
            logging.warning(f"Insufficient training data for {sensor_name}")
            return {'sensor': sensor_name, 'anomalies_count': 0, 'total_points': 0, 'anomaly_rate': 0.0}

        model = SimplePercentileAD(percentiles=sensor_config['threshold_percentiles'])
        model.fit(df_train["_value"].dropna().values)

        df_test = hook.query_to_df(test_query)
        if df_test is None or df_test.empty or df_test["_value"].dropna().shape[0] == 0:
            logging.warning(f"No test data for {sensor_name}")
            return {'sensor': sensor_name, 'anomalies_count': 0, 'total_points': 0, 'anomaly_rate': 0.0}

        values = df_test["_value"].dropna().values
        mask = model.predict(values)
        n_anomalies = int(mask.sum())

        logging.info(f"=== ANOMALY DETECTION: {sensor_name.upper()} ===")
        logging.info(f"Thresholds: {model.thresholds}")
        logging.info(f"Anomalies: {n_anomalies} ({(n_anomalies/len(values))*100:.2f}%)")

        return {
            'sensor': sensor_name,
            'anomalies_count': n_anomalies,
            'total_points': int(len(values)),
            'anomaly_rate': float((n_anomalies/len(values))*100),
            'thresholds': [float(model.thresholds[0]), float(model.thresholds[1])]
        }

    except SSLError as e:
        msg = ("SSLError: revisa Connection 'influxdb' → schema=http y verify_ssl=false "
               "salvo que tengas TLS real en Influx.")
        logging.error(msg)
        return {'sensor': sensor_name, 'anomalies_count': 0, 'error': f"{e}\n{msg}"}
    except Exception as e:
        logging.error(f"Error in anomaly detection for {sensor_name}: {str(e)}")
        return {'sensor': sensor_name, 'anomalies_count': 0, 'error': str(e)}

def consolidate_results(**kwargs):
    ti = kwargs['ti']
    temp_result = ti.xcom_pull(task_ids='detect_temperature_anomalies')
    humidity_result = ti.xcom_pull(task_ids='detect_humidity_anomalies')
    total_anomalies = (temp_result or {}).get('anomalies_count', 0) + (humidity_result or {}).get('anomalies_count', 0)
    logging.info("=== CONSOLIDATED ANOMALY REPORT ===")
    logging.info(f"Temperature anomalies: {(temp_result or {}).get('anomalies_count', 0)}")
    logging.info(f"Humidity anomalies: {(humidity_result or {}).get('anomalies_count', 0)}")
    logging.info(f"Total anomalies: {total_anomalies}")
    if total_anomalies > 0:
        logging.warning(f"ALERT: {total_anomalies} anomalies detected across sensors!")
    return {
        'total_anomalies': total_anomalies,
        'temperature_result': temp_result,
        'humidity_result': humidity_result
    }

with DAG(
    dag_id='anomalydagneo',
    description='Anomaly Detection DAG for Temperature and Humidity (Neo4j-driven, unlabeled-safe)',
    start_date=pendulum.datetime(2025, 9, 20, 0, 0, tz=TZ),
    schedule='*/5 * * * *',
    catchup=False,
    default_args={
        'owner': 'iotlab',
        'retries': 2,
        'retry_delay': timedelta(minutes=2),
        'email_on_failure': False,
        'email_on_retry': False
    },
    tags=['iot', 'influxdb', 'anomaly-detection', 'neo4j', 'dht11']
) as dag:

    load_asset_config = PythonOperator(
        task_id="load_asset_config",
        python_callable=load_asset_config_from_neo4j
    )

    validate = PythonOperator(
        task_id="validate_sources",
        python_callable=validate_sources
    )

    temperature_anomaly_task = PythonOperator(
        task_id="detect_temperature_anomalies",
        python_callable=detect_anomalies_for_sensor,
        op_kwargs={'sensor_name': 'temperature'},
    )

    humidity_anomaly_task = PythonOperator(
        task_id="detect_humidity_anomalies",
        python_callable=detect_anomalies_for_sensor,
        op_kwargs={'sensor_name': 'humidity'},
    )

    consolidate_task = PythonOperator(
        task_id="consolidate_anomaly_results",
        python_callable=consolidate_results
    )

    load_asset_config >> validate >> [temperature_anomaly_task, humidity_anomaly_task] >> consolidate_task

