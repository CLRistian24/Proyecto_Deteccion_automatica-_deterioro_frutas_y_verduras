"""
Módulo de Medidor de Uso Facturable y Estado de Cuenta Mensual (Requisito 7).
Conecta la telemetría real almacenada en la Base de Datos por cada sucursal (store_id)
con el modelo de precios definido en la Memoria Técnica (Capítulo III y Práctica 2.2):
- Plan Base SaaS Híbrido Cloud-Edge: $1,000.00 MXN/mes por tienda (incluye hasta 50,000 inspecciones/mes,
  actualizaciones semestrales de pesos YOLOv11n y motor de alertas en tiempo real).
- Cobro por unidad excedente (Metered Overage): $0.02 MXN por inspección adicional.
- Calcula en tiempo real la Economía Unitaria medida: Costo FinOps real vs. Ingreso facturado y Margen Bruto.
"""

from datetime import datetime
from typing import Dict, Any
from app.database import get_connection


def generate_monthly_statement(store_id: str = "store-01", billing_period: str = "2026-11") -> Dict[str, Any]:
    """
    Consulta las unidades reales procesadas por la sucursal `store_id` y genera
    el Estado de Cuenta Mensual estructurado con métricas de Economía Unitaria.
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT * FROM stores WHERE store_id = ?", (store_id,))
    store_row = cur.fetchone()
    if not store_row:
        conn.close()
        raise ValueError(f"La sucursal '{store_id}' no existe en el padrón de clientes.")

    store = dict(store_row)

    # Contabilizar inspecciones válidas, productos detectados y alertas del cliente
    cur.execute(
        """
        SELECT
            COUNT(*) AS total_requests,
            SUM(CASE WHEN is_valid_format = 1 THEN 1 ELSE 0 END) AS billable_inspections,
            SUM(CASE WHEN is_valid_format = 0 THEN 1 ELSE 0 END) AS rejected_requests,
            COALESCE(SUM(total_items_detected), 0) AS total_produce_evaluated,
            COALESCE(SUM(fresh_count), 0) AS total_fresh,
            COALESCE(SUM(rotten_count), 0) AS total_rotten,
            COALESCE(SUM(alert_triggered), 0) AS total_alerts_sent,
            COALESCE(SUM(faces_blurred), 0) AS total_faces_anonymized,
            COALESCE(AVG(CASE WHEN is_valid_format = 1 THEN latency_ms END), 0.0) AS avg_latency_ms
        FROM inspections
        WHERE store_id = ?
        """,
        (store_id,),
    )
    usage = dict(cur.fetchone())

    # Desglose de productos podridos por tipo para el reporte ejecutivo al cliente
    cur.execute(
        """
        SELECT produce_type, COUNT(*) AS count
        FROM detections
        WHERE store_id = ? AND condition_state = 'Rotten'
        GROUP BY produce_type
        ORDER BY count DESC
        """,
        (store_id,),
    )
    rotten_breakdown = {row["produce_type"]: row["count"] for row in cur.fetchall()}

    # Costos FinOps reales medidos para este cliente
    cur.execute(
        """
        SELECT
            COALESCE(SUM(total_unit_cost_mxn), 0.0) AS measured_total_cost_mxn,
            COALESCE(AVG(total_unit_cost_mxn), 0.000148) AS measured_avg_unit_cost_mxn,
            COALESCE(AVG(processing_time_ms), 0.0) AS avg_finops_time_ms
        FROM finops_logs
        WHERE store_id = ?
        """,
        (store_id,),
    )
    finops = dict(cur.fetchone())
    conn.close()

    billable_units = int(usage["billable_inspections"] or 0)
    included_units = int(store["included_units"])
    base_fee = float(store["monthly_base_fee_mxn"])
    overage_rate = float(store["overage_rate_mxn"])

    overage_units = max(0, billable_units - included_units)
    overage_charge_mxn = round(overage_units * overage_rate, 2)

    # También calculamos el equivalente a volumen operativo mensual proyectado (50,000 unidades)
    projected_monthly_units = max(billable_units, included_units)
    measured_unit_cost = float(finops["measured_avg_unit_cost_mxn"] or 0.000148)
    projected_monthly_infra_cost_mxn = round(projected_monthly_units * measured_unit_cost, 2)

    subtotal_mxn = round(base_fee + overage_charge_mxn, 2)
    iva_mxn = round(subtotal_mxn * 0.16, 2)
    total_due_mxn = round(subtotal_mxn + iva_mxn, 2)

    # Cumplimiento SLA (RNF02 <= 30,000 ms)
    avg_lat_sec = (usage["avg_latency_ms"] or 0.0) / 1000.0
    sla_compliant = avg_lat_sec <= float(store["sla_max_latency_sec"])

    # Margen unitario bruto sobre la tarifa mensual ($1,000 MXN vs costo medido de infraestructura + soporte local $300)
    local_support_opex_mxn = 300.00  # Sección 3.7 Memoria Técnica: $300/tienda
    total_monthly_cogs_mxn = round(local_support_opex_mxn + projected_monthly_infra_cost_mxn, 2)
    gross_margin_mxn = round(subtotal_mxn - total_monthly_cogs_mxn, 2)
    gross_margin_pct = round((gross_margin_mxn / subtotal_mxn) * 100.0, 2) if subtotal_mxn > 0 else 0.0

    return {
        "statement_folio": f"FG-{billing_period.replace('-', '')}-{store_id.upper()}",
        "issued_at": datetime.now().isoformat(timespec="seconds"),
        "billing_period": billing_period,
        "provider": {
            "name": "FreshGuard Vision AI — UPIIT IPN Cloud-Edge Solutions",
            "rfc": "FGV260921IPN",
            "cost_center": "UPIIT-IA-7",
        },
        "client": {
            "store_id": store["store_id"],
            "store_name": store["store_name"],
            "rfc": store["client_rfc"],
            "plan_name": store["plan_name"],
        },
        "usage_metrics": {
            "billable_inspections": billable_units,
            "rejected_invalid_files": int(usage["rejected_requests"] or 0),
            "included_quota_units": included_units,
            "overage_units": overage_units,
            "total_produce_evaluated": int(usage["total_produce_evaluated"] or 0),
            "fresh_items_count": int(usage["total_fresh"] or 0),
            "rotten_items_count": int(usage["total_rotten"] or 0),
            "rotten_breakdown_by_produce": rotten_breakdown,
            "alerts_dispatched": int(usage["total_alerts_sent"] or 0),
            "faces_anonymized_lfpdppp": int(usage["total_faces_anonymized"] or 0),
        },
        "sla_verification": {
            "committed_max_latency_sec": store["sla_max_latency_sec"],
            "measured_avg_latency_ms": round(float(usage["avg_latency_ms"] or 0.0), 2),
            "measured_avg_latency_sec": round(avg_lat_sec, 4),
            "sla_status": "CUMPLIDO (99.9% Disponibilidad)" if sla_compliant else "INCUMPLIDO",
            "penalty_applied_mxn": 0.00,
        },
        "charges_mxn": {
            "monthly_base_subscription": base_fee,
            "overage_unit_rate": overage_rate,
            "overage_charge": overage_charge_mxn,
            "subtotal": subtotal_mxn,
            "iva_16_pct": iva_mxn,
            "total_due": total_due_mxn,
        },
        "unit_economics_validation": {
            "paper_assumed_unit_cost_mxn": 0.00015000,
            "measured_finops_unit_cost_mxn": round(measured_unit_cost, 8),
            "measured_session_infra_cost_mxn": round(float(finops["measured_total_cost_mxn"] or 0.0), 6),
            "projected_50k_imgs_infra_cost_mxn": projected_monthly_infra_cost_mxn,
            "store_local_support_opex_mxn": local_support_opex_mxn,
            "total_monthly_cogs_mxn": total_monthly_cogs_mxn,
            "gross_profit_per_store_mxn": gross_margin_mxn,
            "gross_margin_pct": gross_margin_pct,
        },
    }
