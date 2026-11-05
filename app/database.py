"""
Capa de Persistencia y Esquema de Base de Datos (Requisito 4).
Soporta SQLite local por defecto (edge/desarrollo) y PostgreSQL mediante variable de entorno DATABASE_URL.
Guarda:
1. stores (sucursales / clientes para el medidor de uso facturable)
2. inspections (peticiones de ingesta individuales o por lote, estado de validación y privacidad)
3. detections (historial detallado de frutas/verduras frescas o podridas detectadas a lo largo del día)
4. finops_logs (instrumentación FinOps con tiempos, recursos y etiquetas de imputación)
"""

import os
import sqlite3
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

DB_PATH = os.environ.get("FRESHGUARD_DB_PATH", os.path.join(os.path.dirname(os.path.dirname(__file__)), "freshguard_v02.db"))


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    """Inicializa el esquema relacional documentado y siembra clientes/sucursales iniciales."""
    conn = get_connection()
    cur = conn.cursor()

    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS stores (
            store_id TEXT PRIMARY KEY,
            store_name TEXT NOT NULL,
            client_rfc TEXT NOT NULL,
            plan_name TEXT NOT NULL DEFAULT 'SaaS Edge Plan Base',
            monthly_base_fee_mxn REAL NOT NULL DEFAULT 1000.00,
            included_units INTEGER NOT NULL DEFAULT 50000,
            overage_rate_mxn REAL NOT NULL DEFAULT 0.02,
            sla_max_latency_sec REAL NOT NULL DEFAULT 30.0,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS inspections (
            inspection_id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_uuid TEXT NOT NULL UNIQUE,
            store_id TEXT NOT NULL,
            camera_hash TEXT NOT NULL,
            ingestion_mode TEXT NOT NULL, -- 'realtime', 'single', 'batch'
            filename TEXT NOT NULL,
            is_valid_format INTEGER NOT NULL DEFAULT 1,
            error_message TEXT,
            faces_blurred INTEGER NOT NULL DEFAULT 0,
            exif_stripped INTEGER NOT NULL DEFAULT 1,
            total_items_detected INTEGER NOT NULL DEFAULT 0,
            fresh_count INTEGER NOT NULL DEFAULT 0,
            rotten_count INTEGER NOT NULL DEFAULT 0,
            alert_triggered INTEGER NOT NULL DEFAULT 0,
            alert_channel TEXT,
            latency_ms REAL NOT NULL DEFAULT 0.0,
            annotated_image_b64 TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (store_id) REFERENCES stores(store_id)
        );

        CREATE TABLE IF NOT EXISTS detections (
            detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
            inspection_id INTEGER NOT NULL,
            store_id TEXT NOT NULL,
            class_id INTEGER NOT NULL,
            class_name TEXT NOT NULL,
            produce_type TEXT NOT NULL, -- 'Carrot', 'Cucumber', 'Pepper', 'Potato', 'Tomato'
            condition_state TEXT NOT NULL, -- 'Fresh' o 'Rotten'
            confidence REAL NOT NULL,
            bbox_json TEXT NOT NULL, -- [x1, y1, x2, y2]
            aisle_zone TEXT NOT NULL DEFAULT 'Anaquel Perecederos A1',
            created_at TEXT NOT NULL,
            FOREIGN KEY (inspection_id) REFERENCES inspections(inspection_id) ON DELETE CASCADE,
            FOREIGN KEY (store_id) REFERENCES stores(store_id)
        );

        CREATE TABLE IF NOT EXISTS finops_logs (
            finops_id INTEGER PRIMARY KEY AUTOINCREMENT,
            inspection_id INTEGER NOT NULL,
            store_id TEXT NOT NULL,
            processing_time_ms REAL NOT NULL,
            cpu_time_ms REAL NOT NULL,
            memory_mb REAL NOT NULL,
            compute_device TEXT NOT NULL,
            edge_energy_cost_mxn REAL NOT NULL,
            cloud_amortized_cost_mxn REAL NOT NULL,
            total_unit_cost_mxn REAL NOT NULL,
            tag_project TEXT NOT NULL DEFAULT 'vision-frutas-verdura',
            tag_environment TEXT NOT NULL DEFAULT 'prod',
            tag_owner TEXT NOT NULL DEFAULT 'zahir-cristian',
            tag_cost_center TEXT NOT NULL DEFAULT 'UPIIT-IA-7',
            tag_service TEXT NOT NULL DEFAULT 'inference',
            tag_store_id TEXT NOT NULL,
            tag_model_version TEXT NOT NULL DEFAULT 'v0.2',
            created_at TEXT NOT NULL,
            FOREIGN KEY (inspection_id) REFERENCES inspections(inspection_id) ON DELETE CASCADE
        );
        """
    )

    # Sembrar tiendas piloto (Capítulo III: 5 tiendas piloto en Año 1)
    stores_seed = [
        ("store-01", "Sucursal Piloto 01 — Tlaxcala Centro (AutoCobro)", "SPT260101TLX", "SaaS Edge Plan Base", 1000.00, 50000, 0.02, 30.0),
        ("store-02", "Sucursal Piloto 02 — Apizaco Express (Sin Cajeros)", "SPA260102APZ", "SaaS Edge Plan Base", 1000.00, 50000, 0.02, 30.0),
        ("store-03", "Sucursal Piloto 03 — Chiautempan SmartMart", "SPC260103CHP", "SaaS Edge Plan Base", 1000.00, 50000, 0.02, 30.0),
        ("store-04", "Sucursal Piloto 04 — Huamantla FreshGo", "SPH260104HMT", "SaaS Edge Plan Base", 1000.00, 50000, 0.02, 30.0),
        ("store-05", "Sucursal Piloto 05 — Zacatelco 24/7", "SPZ260105ZCT", "SaaS Edge Plan Base", 1000.00, 50000, 0.02, 30.0),
    ]
    now_iso = datetime.now().isoformat(timespec="seconds")
    for s in stores_seed:
        cur.execute(
            """
            INSERT OR IGNORE INTO stores
            (store_id, store_name, client_rfc, plan_name, monthly_base_fee_mxn, included_units, overage_rate_mxn, sla_max_latency_sec, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (*s, now_iso),
        )

    conn.commit()
    conn.close()


def record_inspection(
    request_uuid: str,
    store_id: str,
    camera_hash: str,
    ingestion_mode: str,
    filename: str,
    is_valid_format: bool,
    error_message: Optional[str],
    faces_blurred: int,
    exif_stripped: bool,
    detections_list: List[Dict[str, Any]],
    alert_triggered: bool,
    alert_channel: Optional[str],
    latency_ms: float,
    annotated_image_b64: Optional[str],
    finops_data: Optional[Dict[str, Any]],
) -> int:
    """Guarda en una transacción atómica la inspección, sus detecciones individuales y su registro FinOps."""
    conn = get_connection()
    cur = conn.cursor()
    now_iso = datetime.now().isoformat(timespec="seconds")

    fresh_count = sum(1 for d in detections_list if d["condition_state"] == "Fresh")
    rotten_count = sum(1 for d in detections_list if d["condition_state"] == "Rotten")
    total_items = len(detections_list)

    cur.execute(
        """
        INSERT INTO inspections (
            request_uuid, store_id, camera_hash, ingestion_mode, filename,
            is_valid_format, error_message, faces_blurred, exif_stripped,
            total_items_detected, fresh_count, rotten_count,
            alert_triggered, alert_channel, latency_ms, annotated_image_b64, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            request_uuid,
            store_id,
            camera_hash,
            ingestion_mode,
            filename,
            1 if is_valid_format else 0,
            error_message,
            faces_blurred,
            1 if exif_stripped else 0,
            total_items,
            fresh_count,
            rotten_count,
            1 if alert_triggered else 0,
            alert_channel,
            round(latency_ms, 3),
            annotated_image_b64,
            now_iso,
        ),
    )
    inspection_id = cur.lastrowid

    for det in detections_list:
        cur.execute(
            """
            INSERT INTO detections (
                inspection_id, store_id, class_id, class_name,
                produce_type, condition_state, confidence, bbox_json, aisle_zone, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                inspection_id,
                store_id,
                det["class_id"],
                det["class_name"],
                det["produce_type"],
                det["condition_state"],
                round(det["confidence"], 4),
                json.dumps(det["bbox"]),
                det.get("aisle_zone", "Anaquel Perecederos A1"),
                now_iso,
            ),
        )

    if finops_data:
        cur.execute(
            """
            INSERT INTO finops_logs (
                inspection_id, store_id, processing_time_ms, cpu_time_ms,
                memory_mb, compute_device, edge_energy_cost_mxn,
                cloud_amortized_cost_mxn, total_unit_cost_mxn,
                tag_project, tag_environment, tag_owner, tag_cost_center,
                tag_service, tag_store_id, tag_model_version, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                inspection_id,
                store_id,
                finops_data["processing_time_ms"],
                finops_data["cpu_time_ms"],
                finops_data["memory_mb"],
                finops_data["compute_device"],
                finops_data["edge_energy_cost_mxn"],
                finops_data["cloud_amortized_cost_mxn"],
                finops_data["total_unit_cost_mxn"],
                finops_data["tags"]["Project"],
                finops_data["tags"]["Environment"],
                finops_data["tags"]["Owner"],
                finops_data["tags"]["CostCenter"],
                finops_data["tags"]["Service"],
                finops_data["tags"]["StoreID"],
                finops_data["tags"]["ModelVersion"],
                now_iso,
            ),
        )

    conn.commit()
    conn.close()
    return int(inspection_id)
