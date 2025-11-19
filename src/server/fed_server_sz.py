import argparse
import base64
import io
import json
import os
import time
from typing import Dict, Any

import numpy as np
import torch
from awscrt import mqtt
from awsiot import mqtt_connection_builder

from src.models.simple_gru import SimpleGRU


# ================================================================
# ENCODING HELPERS (MATCH CLIENT)
# ================================================================

def encode_state_dict(state_dict: Dict[str, Any]) -> str:
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def decode_state_dict(b64: str) -> Dict[str, Any]:
    raw = base64.b64decode(b64)
    buf = io.BytesIO(raw)
    buf.seek(0)
    return torch.load(buf, map_location="cpu")


# ================================================================
# FEDAVG (UNWEIGHTED) AGGREGATION
# ================================================================

def fedavg(states: Dict[int, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Simple FedAvg: uniform average over all client updates.
    """
    if not states:
        raise ValueError("No client states to aggregate.")

    client_ids = list(states.keys())
    k0 = client_ids[0]
    result = {}

    for name, tensor in states[k0].items():
        acc = tensor.clone().float()
        for cid in client_ids[1:]:
            acc += states[cid][name].float()
        acc /= float(len(client_ids))
        result[name] = acc

    return result


# ================================================================
# SERVER CLASS
# ================================================================

class FederatedServerSZ:
    def __init__(
        self,
        endpoint: str,
        cert_path: str,
        key_path: str,
        root_ca_path: str,
        num_clients: int = 5,
        rounds: int = 3,
        proc_dir: str = "data/processed/sz/prepared",
    ):
        self.endpoint = endpoint
        self.cert_path = cert_path
        self.key_path = key_path
        self.root_ca_path = root_ca_path
        self.num_clients = num_clients
        self.rounds = rounds
        self.proc_dir = proc_dir

        # MQTT topics (must match client config)
        self.topic_round_start = "aefl/fl/sz/round/start"
        self.topic_global_model = "aefl/fl/sz/global_model"
        self.topic_client_update = "aefl/fl/sz/client_update/+"
        self.topic_stop = "aefl/fl/sz/stop"

        self.mqtt_connection = None
        self.global_state = None
        self.client_updates: Dict[int, Dict[str, torch.Tensor]] = {}

    # -----------------------------
    # DATA + MODEL INIT
    # -----------------------------

    def _infer_num_nodes(self) -> int:
        X_path = os.path.join(self.proc_dir, "X_train.npy")
        if not os.path.exists(X_path):
            raise FileNotFoundError(f"Missing {X_path}. Run data.preprocess_sz first.")
        X = np.load(X_path)
        return X.shape[-1]

    def init_model(self):
        num_nodes = self._infer_num_nodes()
        print(f"[SERVER] SZ num_nodes = {num_nodes}")
        model = SimpleGRU(num_nodes=num_nodes, hidden_size=64)
        self.global_state = model.state_dict()

    # -----------------------------
    # MQTT
    # -----------------------------

    def connect_mqtt(self):
        print(f"[SERVER] Connecting to AWS IoT at {self.endpoint} ...")
        self.mqtt_connection = mqtt_connection_builder.mtls_from_path(
            endpoint=self.endpoint,
            cert_filepath=self.cert_path,
            pri_key_filepath=self.key_path,
            ca_filepath=self.root_ca_path,
            client_id="AEFL_Server",
            clean_session=True,
            keep_alive_secs=60,
        )
        self.mqtt_connection.connect().result()
        print("[SERVER] Connected to IoT Core.")

        # Subscribe to all client_update topics
        print(f"[SERVER] Subscribing to: {self.topic_client_update}")
        self.mqtt_connection.subscribe(
            topic=self.topic_client_update,
            qos=mqtt.QoS.AT_LEAST_ONCE,
            callback=self._on_client_update,
        )

    def _on_client_update(self, topic, payload, **kwargs):
        try:
            msg = json.loads(payload.decode("utf-8"))
            cid = int(msg["client_id"])
            state_b64 = msg["state_dict"]
            state = decode_state_dict(state_b64)
            self.client_updates[cid] = state
            print(f"[SERVER] Received update from client {cid}. "
                  f"Total updates: {len(self.client_updates)}/{self.num_clients}")
        except Exception as e:
            print(f"[SERVER] ERROR parsing client update: {e}")

    # -----------------------------
    # ONE ROUND
    # -----------------------------

    def run_round(self, r: int, wait_timeout: float = 300.0):
        """
        Run a single FL round:
          1) broadcast global model
          2) signal round start
          3) wait for client updates
          4) aggregate
        """
        print(f"\n[SERVER] ========== ROUND {r} ==========")
        self.client_updates.clear()

        # 1) Broadcast global model
        model_msg = {
            "round": r,
            "state_dict": encode_state_dict(self.global_state),
        }
        self.mqtt_connection.publish(
            topic=self.topic_global_model,
            payload=json.dumps(model_msg).encode("utf-8"),
            qos=mqtt.QoS.AT_LEAST_ONCE,
        )
        print("[SERVER] Published GLOBAL_MODEL.")

        # 2) Signal round start (clients set round_in_progress flag)
        self.mqtt_connection.publish(
            topic=self.topic_round_start,
            payload=json.dumps({"round": r}).encode("utf-8"),
            qos=mqtt.QoS.AT_LEAST_ONCE,
        )
        print("[SERVER] Published ROUND_START.")

        # 3) Wait for client updates
        t0 = time.time()
        while len(self.client_updates) < self.num_clients:
            if time.time() - t0 > wait_timeout:
                print("[SERVER] WAIT TIMEOUT. Proceeding with available updates.")
                break
            time.sleep(2.0)

        if not self.client_updates:
            print("[SERVER] No updates received. Aborting training.")
            return False

        print(f"[SERVER] Aggregating {len(self.client_updates)} client updates...")

        # 4) FedAvg aggregation
        self.global_state = fedavg(self.client_updates)

        return True

    # -----------------------------
    # MAIN RUN
    # -----------------------------

    def run(self):
        self.init_model()
        self.connect_mqtt()

        os.makedirs("outputs/server", exist_ok=True)

        for r in range(1, self.rounds + 1):
            ok = self.run_round(r)
            if not ok:
                break
            # Optional: save intermediate global model
            torch.save(
                self.global_state,
                f"outputs/server/global_round_{r}.pt"
            )
            print(f"[SERVER] Saved global model for round {r}.")

        # After rounds, tell all clients to stop
        print("[SERVER] Training complete. Sending STOP signal...")
        self.mqtt_connection.publish(
            topic=self.topic_stop,
            payload=json.dumps({"reason": "done"}).encode("utf-8"),
            qos=mqtt.QoS.AT_LEAST_ONCE,
        )

        time.sleep(2.0)
        self.mqtt_connection.disconnect().result()
        print("[SERVER] Disconnected from IoT Core. Done.")


# ================================================================
# CLI ENTRYPOINT
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="AEFL Federated Server for SZ dataset")
    parser.add_argument("--endpoint", type=str, required=True, help="AWS IoT Core ATS endpoint")
    parser.add_argument("--cert", type=str, required=True, help="Server certificate path")
    parser.add_argument("--key", type=str, required=True, help="Server private key path")
    parser.add_argument("--root-ca", type=str, required=True, help="Root CA path (AmazonRootCA1.pem)")
    parser.add_argument("--rounds", type=int, default=3, help="Number of FL rounds")
    parser.add_argument("--num-clients", type=int, default=5, help="Number of participating clients")
    parser.add_argument("--proc-dir", type=str, default="data/processed/sz/prepared", help="Processed SZ dir")

    args = parser.parse_args()

    server = FederatedServerSZ(
        endpoint=args.endpoint,
        cert_path=args.cert,
        key_path=args.key,
        root_ca_path=args.root_ca,
        num_clients=args.num_clients,
        rounds=args.rounds,
        proc_dir=args.proc_dir,
    )
    server.run()


if __name__ == "__main__":
    main()
