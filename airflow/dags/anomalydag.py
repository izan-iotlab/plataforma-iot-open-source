import pendulum
from datetime import timedelta
import logging

from airflow.models.dag import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.influxdb.hooks.influxdb import InfluxDBHook

SENSORS_CONFIG = {
    'temperature': {
        'topic': 'empresaA/barcelona/linea1/hornoA/telemetry/temperature',
        'measurement': 'temperature',
        'threshold_percentiles': (1, 99),
        'window_minutes': 5
    },
    'humidity': {
        'topic': 'empresaA/barcelona/linea1/hornoA/telemetry/humidity',
        'measurement': 'humidity',
        'threshold_percentiles': (5, 95),
        'window_minutes': 5
    }
}

TZ = "Europe/Madrid"

def rfc3339_utc(dt):
    # Tiempos en UTC con Z para Flux
    return pendulum.instance(dt).in_timezone('UTC').to_iso8601_string().replace('+00:00', 'Z')

def get_sensor_query(sensor_config, start_time, stop_time):
    # start_time/stop_time deben llegar en RFC3339 (ej. 2025-09-21T09:35:00Z)
    return f'''
from(bucket: "bucket0")
  |> range(start: time(v: "{start_time}"), stop: time(v: "{stop_time}"))
  |> filter(fn: (r) => r["_measurement"] == "{sensor_config['measurement']}")
  |> filter(fn: (r) => r["topic"] == "{sensor_config['topic']}")
  |> filter(fn: (r) => r["_field"] == "value")
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

def detect_anomalies_for_sensor(sensor_name, **kwargs):
    import numpy as np
    from requests.exceptions import SSLError

    sensor_config = SENSORS_CONFIG[sensor_name]
    hook = InfluxDBHook(conn_id="influxdb")

    now_local = pendulum.now(TZ)
    train_start = now_local - timedelta(hours=1)
    train_stop  = now_local - timedelta(minutes=sensor_config['window_minutes'])
    test_start  = now_local - timedelta(minutes=sensor_config['window_minutes'])
    test_stop   = now_local

    # → Pasamos UTC-Z y usamos time(v: "...") en la query
    train_query = get_sensor_query(sensor_config, rfc3339_utc(train_start), rfc3339_utc(train_stop))
    test_query  = get_sensor_query(sensor_config, rfc3339_utc(test_start),  rfc3339_utc(test_stop))

    try:
        df_train = hook.query_to_df(train_query)
        if df_train.empty or df_train["_value"].dropna().shape[0] < 10:
            logging.warning(f"Insufficient training data for {sensor_name}")
            return {'sensor': sensor_name, 'anomalies_count': 0, 'total_points': 0, 'anomaly_rate': 0.0}

        model = SimplePercentileAD(percentiles=sensor_config['threshold_percentiles'])
        model.fit(df_train["_value"].dropna().values)

        df_test = hook.query_to_df(test_query)
        if df_test.empty or df_test["_value"].dropna().shape[0] == 0:
            logging.warning(f"No test data for {sensor_name}")
            return {'sensor': sensor_name, 'anomalies_count': 0, 'total_points': 0, 'anomaly_rate': 0.0}

        values = df_test["_value"].dropna().values
        mask = model.predict(values)
        n_anomalies = int(mask.sum())

        logging.info(f"=== ANOMALY DETECTION: {sensor_name.upper()} ===")
        logging.info(f"Training points: {len(df_train)} | Test points: {len(df_test)}")
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

with DAG(
    dag_id='anomalydag',
    description='Anomaly Detection DAG for Temperature and Humidity Sensors',
    start_date=pendulum.datetime(2025, 9, 26, 0, 0, tz=TZ),
    schedule='*/5 * * * *',
    catchup=False,
    default_args={
        'owner': 'iotlab',
        'retries': 2,
        'retry_delay': timedelta(minutes=2),
        'email_on_failure': False,
        'email_on_retry': False
    },
    tags=['iot', 'influxdb', 'anomaly-detection', 'dht11']
) as dag:

    temperature_anomaly_task = PythonOperator(
        task_id="detect_temperature_anomalies",
        python_callable=detect_anomalies_for_sensor,
        op_kwargs={'sensor_name': 'temperature'},
        pool='sensor_analysis_pool',
        pool_slots=1
    )

    humidity_anomaly_task = PythonOperator(
        task_id="detect_humidity_anomalies",
        python_callable=detect_anomalies_for_sensor,
        op_kwargs={'sensor_name': 'humidity'},
        pool='sensor_analysis_pool',
        pool_slots=1
    )

    def consolidate_results(**kwargs):
        import logging
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

    consolidate_task = PythonOperator(
        task_id="consolidate_anomaly_results",
        python_callable=consolidate_results
    )

    [temperature_anomaly_task, humidity_anomaly_task] >> consolidate_task
