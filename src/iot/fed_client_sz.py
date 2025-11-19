import os
import json
import time
import base64
import io

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from awscrt import mqtt
from awsiot import mqtt_connection_builder

# ================================================================
# SAFE ENVIRONMENT LOADING
# ================================================================

def require_env(name):
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val

CLIENT_ID = int(os.environ.get("CLIENT_ID", 0))

IOT_ENDPOINT = require_env("IOT_ENDPOINT")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
S3_BUCKET = os.environ.get("S3_BUCKET", "aefl-results")
S3_PREFIX = os.environ.get("S3_PREFIX", "fl-sz")

CERT_PATH = require_env("CERT_PATH")
KEY_PATH = require_env("KEY_PATH")
ROOT_CA_PATH = require_env("ROOT_CA_PATH")

PROC_DIR = os.environ.get("PROC_DIR", "data/processed/sz/prepared")

LOCAL_EPOCHS = int(os.environ.get("LOCAL_EPOCHS", 1))
LR = float(os.environ.get("LR", 0.001))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 64))

DEVICE = "cpu"


# MQTT topics
TOPIC_ROUND_START = "aefl/fl/sz/round/start"
TOPIC_GLOBAL_MODEL = "aefl/fl/sz/global_model"
TOPIC_CLIENT_UPDATE = f"aefl/fl/sz/client_update/{CLIENT_ID}"
TOPIC_STOP = "aefl/fl/sz/stop"


# ================================================================
# DATA LOADING
# ================================================================
def load_local_data():
    x_path = os.path.join(PROC_DIR, "clients", f"client{CLIENT_ID}_X.npy")
    y_path = os.path.join(PROC_DIR, "clients", f"client{CLIENT_ID}_y.npy")

    if not os.path.exists(x_path):
        raise FileNotFoundError(f"Missing {x_path}")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"Missing {y_path}")

    X = np.load(x_path)
    y = np.load(y_path)

    ds = TensorDataset(torch.from_numpy(X).float(),
                       torch.from_numpy(y).float())
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)


# ================================================================
# MODEL
# ================================================================
class SimpleGRU(torch.nn.Module):
    def __init__(self, num_nodes, hidden_size=64):
        super().__init__()
        self.gru = torch.nn.GRU(
            input_size=num_nodes,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.head = torch.nn.Linear(hidden_size, num_nodes)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


# ================================================================
# SERIALIZATION
# ================================================================
def encode_state_dict(state_dict):
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def decode_state_dict(b64):
    raw = base64.b64decode(b64)
    buf = io.BytesIO(raw)
    buf.seek(0)
    return torch.load(buf, map_location="cpu")


# ================================================================
# MQTT CONNECTION
# ================================================================
mqtt_connection = None

def connect_mqtt():
    global mqtt_connection
    print(f"[Client {CLIENT_ID}] Connecting to AWS IoT Core at {IOT_ENDPOINT} ...")

    mqtt_connection = mqtt_connection_builder.mtls_from_path(
        endpoint=IOT_ENDPOINT,
        cert_filepath=CERT_PATH,
        pri_key_filepath=KEY_PATH,
        ca_filepath=ROOT_CA_PATH,
        client_id=f"aefl_client_{CLIENT_ID}",
        clean_session=True,
        keep_alive_secs=30
    )

    mqtt_connection.connect().result()
    print(f"[Client {CLIENT_ID}] Connected.")
    return mqtt_connection


# ================================================================
# FLAGS
# ================================================================
round_in_progress = False
global_model_state = None
stop_flag = False


# ================================================================
# CALLBACKS
# ================================================================
def on_round_start(topic=None, payload=None):
    global round_in_progress
    round_in_progress = True
    print(f"[Client {CLIENT_ID}] ROUND_START received.")

def on_global_model(topic=None, payload=None):
    global global_model_state
    try:
        msg = payload.decode("utf-8")
        state_b64 = json.loads(msg)["state_dict"]
        global_model_state = decode_state_dict(state_b64)
        print(f"[Client {CLIENT_ID}] GLOBAL_MODEL received.")
    except Exception as e:
        print(f"[Client {CLIENT_ID}] Error decoding global model: {e}")

def on_stop(topic=None, payload=None):
    global stop_flag
    stop_flag = True
    print(f"[Client {CLIENT_ID}] STOP received.")


# AWS IoT requires wrapper signature
def _cb_round_start(topic, payload, **kwargs):
    on_round_start(topic=topic, payload=payload)

def _cb_global_model(topic, payload, **kwargs):
    on_global_model(topic=topic, payload=payload)

def _cb_stop(topic, payload, **kwargs):
    on_stop(topic=topic, payload=payload)


# ================================================================
# TRAINING
# ================================================================
def train_one_round(model, loader):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    for _ in range(LOCAL_EPOCHS):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()

    return {k: v.cpu() for k, v in model.state_dict().items()}


# ================================================================
# SEND UPDATE
# ================================================================
def send_update(state_dict):
    msg = json.dumps({
        "client_id": CLIENT_ID,
        "state_dict": encode_state_dict(state_dict),
    })

    mqtt_connection.publish(
        topic=TOPIC_CLIENT_UPDATE,
        payload=msg.encode("utf-8"),
        qos=mqtt.QoS.AT_LEAST_ONCE
    )

    print(f"[Client {CLIENT_ID}] SENT update.")


# ================================================================
# MAIN LOOP
# ================================================================
def main():
    global global_model_state, round_in_progress, stop_flag

    loader = load_local_data()
    conn = connect_mqtt()

    # Subscribe
    conn.subscribe(topic=TOPIC_ROUND_START, qos=mqtt.QoS.AT_LEAST_ONCE, callback=_cb_round_start)
    conn.subscribe(topic=TOPIC_GLOBAL_MODEL, qos=mqtt.QoS.AT_LEAST_ONCE, callback=_cb_global_model)
    conn.subscribe(topic=TOPIC_STOP, qos=mqtt.QoS.AT_LEAST_ONCE, callback=_cb_stop)

    model = None
    print(f"[Client {CLIENT_ID}] READY.")

    while not stop_flag:
        if round_in_progress and global_model_state:
            round_in_progress = False

            if model is None:
                x, _ = next(iter(loader))
                N = x.shape[-1]
                model = SimpleGRU(num_nodes=N).to(DEVICE)

            try:
                model.load_state_dict(global_model_state)
            except Exception as e:
                print(f"[Client {CLIENT_ID}] Failed loading state: {e}")
                global_model_state = None
                continue

            updated = train_one_round(model, loader)
            send_update(updated)

            global_model_state = None

        time.sleep(0.3)

    print(f"[Client {CLIENT_ID}] EXIT.")


if __name__ == "__main__":
    main()
