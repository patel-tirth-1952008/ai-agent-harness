# agent1_portfolio_builder.py
import os
import sys
import json
import time
import re
import random
import subprocess
import requests
import base64
from datetime import datetime

# Monkeypatch CrewAI / LiteLLM bug: remove cache_breakpoint and cache_control from messages to prevent HTTP 400
try:
    import litellm
    _orig_completion = litellm.completion
    _orig_acompletion = litellm.acompletion

    def _clean_messages(messages):
        cleaned = []
        for msg in messages:
            if not isinstance(msg, dict):
                cleaned.append(msg)
                continue
            new_msg = {k: v for k, v in msg.items() if k not in ["cache_breakpoint", "cache_control"]}
            cleaned.append(new_msg)
        return cleaned

    def _patched_completion(*args, **kwargs):
        if "messages" in kwargs:
            kwargs["messages"] = _clean_messages(kwargs["messages"])
        if "LITELLM_DROP_PARAMS" not in os.environ:
            os.environ["LITELLM_DROP_PARAMS"] = "True"
        kwargs["num_retries"] = 5
        kwargs["retry_delay"] = 35.0
        return _orig_completion(*args, **kwargs)

    def _patched_acompletion(*args, **kwargs):
        if "messages" in kwargs:
            kwargs["messages"] = _clean_messages(kwargs["messages"])
        if "LITELLM_DROP_PARAMS" not in os.environ:
            os.environ["LITELLM_DROP_PARAMS"] = "True"
        kwargs["num_retries"] = 5
        kwargs["retry_delay"] = 35.0
        return _orig_acompletion(*args, **kwargs)

    litellm.completion = _patched_completion
    litellm.acompletion = _patched_acompletion
except Exception as e:
    print(f"Warning: LiteLLM patch failed to initialize: {e}")

# Safe Backtick constant to avoid breaking chat output parsed formats
B3 = chr(96) * 3

# GLOBAL AGENT CONFIGURATION
MODEL_NAME = "groq/qwen/qwen3.8-27b"
MAX_TOKENS = 950
STAGE_DELAY = 65
REVIEW_DELAY = 65
MAX_REVIEW_CYCLES = 3

PORTFOLIO_HISTORY_FILE = "data/portfolio_history.json"

BLUEPRINTS = [
    {
        "name": "cloudops-sentinel",
        "title": "CloudOps Sentinel - Real-Time Infrastructure Health & Incident Monitor",
        "category": "DevOps & Cloud Infrastructure",
        "description": "Production-grade microservices health monitor and automated incident alerting engine with latency anomaly detection, uptime SLAs, and webhook notifications.",
        "tech_stack": "FastAPI, Next.js 14, TailwindCSS, Docker Compose",
        "backend_endpoints": [
            {"method": "GET", "path": "/api/services", "desc": "List all monitored infrastructure services and status"},
            {"method": "POST", "path": "/api/services", "desc": "Register a new microservice endpoint for health tracking"},
            {"method": "GET", "path": "/api/incidents", "desc": "Retrieve active and historical service incidents"},
            {"method": "POST", "path": "/api/incidents/{incident_id}/resolve", "desc": "Mark an active alert/incident as resolved"},
            {"method": "GET", "path": "/api/metrics/sla", "desc": "Calculate live SLA uptime percentages per service"},
            {"method": "POST", "path": "/api/ping", "desc": "Ingest real-time health check heartbeat reports"},
            {"method": "GET", "path": "/api/health", "desc": "System self-health check"}
        ],
        "db_schema": [
            "id: str (UUID)",
            "name: str",
            "url: str",
            "status: str (healthy/degraded/down)",
            "latency_ms: int",
            "last_checked: datetime",
            "uptime_percentage: float"
        ],
        "sample_data": [
            {"id": "s1", "name": "Payment Gateway API", "url": "https://api.payments.internal/health", "status": "healthy", "latency_ms": 42, "uptime_percentage": 99.98},
            {"id": "s2", "name": "Auth Core Identity Server", "url": "https://auth.internal/v1/ping", "status": "degraded", "latency_ms": 420, "uptime_percentage": 99.85},
            {"id": "s3", "name": "Legacy Report Exporter", "url": "https://reports.internal/status", "status": "down", "latency_ms": 0, "uptime_percentage": 94.20}
        ],
        "frontend_sections": [
            "Infrastructure Status Grid (Visual badges, live latency chart indicators, total stats summary)",
            "Active Incident Alert Banner (Red pulsing triggers, SLA degradation warnings, resolve buttons)",
            "Register New Service form (Interactive fields, URL validation, real-time feedback)",
            "Comprehensive Endpoint Details Modal (Historical uptime metrics, latency timeline graphs)"
        ],
        "test_cases": [
            "test_get_all_services_returns_prepopulated_sample_data",
            "test_register_new_service_validates_url_schema",
            "test_ping_updates_service_latency_and_timestamp",
            "test_ping_trigger_status_degraded_when_latency_high",
            "test_incident_created_when_service_goes_down",
            "test_resolve_incident_updates_status_to_healthy",
            "test_sla_calculation_ignores_negative_or_invalid_values",
            "test_health_endpoint_returns_healthy_status"
        ]
    },
    {
        "name": "rag-doc-intel",
        "title": "RAG Document Intelligence & Semantic Search Hub",
        "category": "Artificial Intelligence & LLMs",
        "description": "High-performance semantic analysis portal with document text parsing, chunk embedding vector simulator, similarity threshold tuning, and citation matching.",
        "tech_stack": "FastAPI, Next.js 14, TailwindCSS, Docker Compose",
        "backend_endpoints": [
            {"method": "GET", "path": "/api/documents", "desc": "List all uploaded text corpora and status"},
            {"method": "POST", "path": "/api/documents", "desc": "Upload and parse document text, simulating auto-vectorization"},
            {"method": "POST", "path": "/api/search", "desc": "Perform simulated semantic query with cosine distance scores"},
            {"method": "GET", "path": "/api/citations/{doc_id}", "desc": "Retrieve highlighted text segments and references"},
            {"method": "DELETE", "path": "/api/documents/{doc_id}", "desc": "De-index document and purge chunks"},
            {"method": "GET", "path": "/api/health", "desc": "System self-health check"}
        ],
        "db_schema": [
            "id: str (UUID)",
            "title: str",
            "content: str",
            "chunk_count: int",
            "upload_time: datetime",
            "embedding_simulated: bool"
        ],
        "sample_data": [
            {"id": "d1", "title": "Corporate Terms of Service.pdf", "content": "Liability limits are capped at $50,000 for standard enterprise tenants, governed by Delaware law.", "chunk_count": 12, "upload_time": "2024-03-01T10:00:00Z", "embedding_simulated": True},
            {"id": "d2", "title": "HIPAA Compliance Guide 2024.docx", "content": "All protected patient health information (PHI) must be stored using AES-256 equivalent server-side encryption.", "chunk_count": 45, "upload_time": "2024-03-02T11:15:00Z", "embedding_simulated": True}
        ],
        "frontend_sections": [
            "Dynamic Semantic Query Input (With sliding relevance confidence filters)",
            "Source Citations Panel (Interactive highlighted text boxes linked to origin PDFs)",
            "Document Knowledge Base Index (Displays chunk counts, token usage, indexing badges)",
            "Simulation Vector Sandbox (Visual projection representing high-dimensional chunk mapping)"
        ],
        "test_cases": [
            "test_get_documents_returns_prepopulated_list",
            "test_upload_document_creates_valid_chunks",
            "test_semantic_search_ranks_by_score_correctly",
            "test_semantic_search_respects_threshold_limits",
            "test_get_citations_returns_valid_content_blocks",
            "test_delete_document_removes_all_chunks_from_index",
            "test_empty_query_returns_error_code_400",
            "test_health_endpoint_is_active"
        ]
    },
    {
        "name": "ratelimit-gateway",
        "title": "API Gateway Rate Limiter & Security Shields Monitor",
        "category": "Cybersecurity & Backend Middleware",
        "description": "Simulated reverse proxy edge protection engine executing sliding-window limiters, IP range blocking, token buckets, and threat classification.",
        "tech_stack": "FastAPI, Next.js 14, TailwindCSS, Docker Compose",
        "backend_endpoints": [
            {"method": "GET", "path": "/api/rules", "desc": "List current rate limit profiles and security policies"},
            {"method": "POST", "path": "/api/rules", "desc": "Add or update security rate limit thresholds"},
            {"method": "POST", "path": "/api/shield/request", "desc": "Ingest client connection, returns 200 OK or 429 Limit Blocked"},
            {"method": "GET", "path": "/api/shield/metrics", "desc": "Calculate real-time drop metrics and block ratios"},
            {"method": "POST", "path": "/api/shield/blacklist", "desc": "Manually ban rogue actor IP blocks"},
            {"method": "GET", "path": "/api/health", "desc": "System self-health check"}
        ],
        "db_schema": [
            "id: str",
            "client_ip: str",
            "endpoint_path: str",
            "request_count: int",
            "window_start: datetime",
            "is_blocked: bool"
        ],
        "sample_data": [
            {"id": "r1", "client_ip": "198.51.100.42", "endpoint_path": "/api/checkout", "request_count": 124, "window_start": "2024-03-03T14:30:00Z", "is_blocked": True},
            {"id": "r2", "client_ip": "203.0.113.88", "endpoint_path": "/api/products", "request_count": 8, "window_start": "2024-03-03T14:31:00Z", "is_blocked": False}
        ],
        "frontend_sections": [
            "Live IP Blocking Dashboard (Animated traffic streams showing Allow vs Block packets)",
            "Dynamic Rules Configurator (Editable thresholds for rate limiting with instant deployment flags)",
            "Active Blacklist Registry (IP pattern grids with release-ban actions)",
            "Threat Logs feed (Pulsing log lines of unauthorized requests and high-frequency alerts)"
        ],
        "test_cases": [
            "test_get_rules_returns_configured_limits",
            "test_request_under_limit_is_allowed",
            "test_request_exceeding_limit_returns_429",
            "test_sliding_window_resets_after_expiry",
            "test_blacklisted_ip_instantly_blocked_with_status_403",
            "test_add_new_limiter_rule_persists_successfully",
            "test_metrics_calculates_correct_block_ratio",
            "test_health_returns_gateway_status"
        ]
    },
    {
        "name": "event-task-engine",
        "title": "Event-Driven Background Task Engine",
        "category": "Distributed Systems & Queues",
        "description": "Background worker dashboard executing persistent asynchronous tasks, retry logic queues, backoff simulations, and dead-letter queue audits.",
        "tech_stack": "FastAPI, Next.js 14, TailwindCSS, Docker Compose",
        "backend_endpoints": [
            {"method": "GET", "path": "/api/tasks", "desc": "List all active, pending, failed, and completed workers"},
            {"method": "POST", "path": "/api/tasks", "desc": "Enqueue a new simulated long-running process"},
            {"method": "POST", "path": "/api/tasks/{task_id}/cancel", "desc": "Interrupt and abort a running worker process"},
            {"method": "GET", "path": "/api/tasks/dlq", "desc": "Retrieve dead-letter queue tasks for debugging"},
            {"method": "POST", "path": "/api/tasks/dlq/retry", "desc": "Flush and re-enqueue all items in dead-letter state"},
            {"method": "GET", "path": "/api/health", "desc": "System self-health check"}
        ],
        "db_schema": [
            "id: str (UUID)",
            "name: str",
            "status: str (queued/processing/completed/failed)",
            "progress: int (0-100)",
            "retry_count: int",
            "payload: str",
            "error_log: str"
        ],
        "sample_data": [
            {"id": "t1", "name": "Compress Media Archive", "status": "processing", "progress": 45, "retry_count": 0, "payload": "assets_v2.tar.gz", "error_log": ""},
            {"id": "t2", "name": "Sync Stripe Subscriptions", "status": "completed", "progress": 100, "retry_count": 1, "payload": "period_march", "error_log": ""},
            {"id": "t3", "name": "Send Bulk Email Notification", "status": "failed", "progress": 12, "retry_count": 3, "payload": "campaign_newsletter", "error_log": "SMTP connection timed out on port 587 after 3 attempts."}
        ],
        "frontend_sections": [
            "Worker Orchestration Hub (Action buttons to enqueue tasks, cancel running, flush DLQ)",
            "Active Job Progress Matrix (Live progress bar updates, status colors, run durations)",
            "Dead-Letter Queue Auditor (Special red panel displaying failures with stack traces and retry levers)",
            "System Queue Metrics Summary (Counter widgets for total queued, completed, failed, and throughput)"
        ],
        "test_cases": [
            "test_get_tasks_returns_active_jobs",
            "test_enqueue_task_creates_job_in_queued_state",
            "test_worker_transitions_to_completed_with_100_percent_progress",
            "test_failed_task_automatically_retries_until_limit",
            "test_task_exceeding_retry_limit_sent_to_dlq",
            "test_cancel_task_aborts_and_updates_status",
            "test_dlq_retry_flushes_failed_tasks_back_to_queue",
            "test_health_check_operational"
        ]
    },
    {
        "name": "featureflag-nexus",
        "title": "Feature Flag Control Center & Targeting Platform",
        "category": "Enterprise SaaS Platform Tools",
        "description": "High-fidelity configuration management center allowing real-time multi-environment feature switching, user percentage targeting rules, and audit logs.",
        "tech_stack": "FastAPI, Next.js 14, TailwindCSS, Docker Compose",
        "backend_endpoints": [
            {"method": "GET", "path": "/api/flags", "desc": "List all registered feature keys and configurations"},
            {"method": "POST", "path": "/api/flags", "desc": "Create a new multi-targeting toggle switch"},
            {"method": "PUT", "path": "/api/flags/{flag_key}", "desc": "Update flag rules, parameters, and active overrides"},
            {"method": "POST", "path": "/api/flags/evaluate", "desc": "Query flag state for a specific user ID with context attributes"},
            {"method": "GET", "path": "/api/audit-logs", "desc": "List user access controls and system mutation histories"},
            {"method": "GET", "path": "/api/health", "desc": "System self-health check"}
        ],
        "db_schema": [
            "key: str",
            "name: str",
            "is_enabled: bool",
            "rollout_percentage: int (0-100)",
            "targeting_rules: str (JSON text)",
            "environment: str (prod/staging)",
            "updated_by: str"
        ],
        "sample_data": [
            {"key": "new-billing-flow", "name": "Stripe Elements checkout UI", "is_enabled": True, "rollout_percentage": 25, "targeting_rules": '{"tier": "enterprise"}', "environment": "prod", "updated_by": "alex@corp.com"},
            {"key": "beta-dark-mode", "name": "Tailwind dark color palettes", "is_enabled": False, "rollout_percentage": 0, "targeting_rules": '{"is_beta_tester": true}', "environment": "staging", "updated_by": "clara@corp.com"}
        ],
        "frontend_sections": [
            "Feature Flag Toggle Grid (Environment-separated toggle cards with percentage sliders)",
            "Context Evaluator Console (Interactive sidebar to input user profiles and check flag status)",
            "Rules Editor Pane (Visual target rules builder with key-value property matching)",
            "Audit Trail Chronology (List of flag updates, rollback actions, modifier metrics)"
        ],
        "test_cases": [
            "test_get_flags_returns_current_switches",
            "test_create_flag_persists_with_correct_rules",
            "test_evaluate_flag_returns_true_when_globally_on",
            "test_evaluate_flag_returns_false_when_globally_off",
            "test_evaluate_flag_handles_percentage_hash_rollout",
            "test_evaluate_flag_resolves_attributes_targeting",
            "test_audit_log_created_on_flag_state_toggle",
            "test_health_returns_nexus_uptime"
        ]
    },
    {
        "name": "prompt-eval-hub",
        "title": "LLM Prompt Engineering Evaluation & Metrics Hub",
        "category": "Artificial Intelligence & LLMs",
        "description": "Continuous prompt optimization dashboard that compares model response outputs, latency profiles, token counts, and semantic quality scores.",
        "tech_stack": "FastAPI, Next.js 14, TailwindCSS, Docker Compose",
        "backend_endpoints": [
            {"method": "GET", "path": "/api/prompts", "desc": "List all active prompt templates and versions"},
            {"method": "POST", "path": "/api/prompts", "desc": "Register a new prompt layout or revision"},
            {"method": "POST", "path": "/api/evaluations", "desc": "Trigger and score simulated outputs against metric suites"},
            {"method": "GET", "path": "/api/metrics/compare", "desc": "Fetch run metadata comparing performance indicators across configurations"},
            {"method": "GET", "path": "/api/health", "desc": "System self-health check"}
        ],
        "db_schema": [
            "id: str (UUID)",
            "prompt_name: str",
            "version: int",
            "system_instruction: str",
            "user_template: str",
            "avg_latency_ms: float",
            "semantic_score: float (0.0-1.0)"
        ],
        "sample_data": [
            {"id": "p1", "prompt_name": "RAG Conversational Agent", "version": 1, "system_instruction": "Answer questions using the provided context.", "user_template": "Context: {context}\nQuestion: {query}", "avg_latency_ms": 1120.5, "semantic_score": 0.88},
            {"id": "p2", "prompt_name": "RAG Conversational Agent", "version": 2, "system_instruction": "You are a concise enterprise legal assistant. Use cited contexts only.", "user_template": "Context: {context}\nQuestion: {query}", "avg_latency_ms": 940.2, "semantic_score": 0.95}
        ],
        "frontend_sections": [
            "Template Comparison Panel (Side-by-side markdown comparison with active code highlighting)",
            "Metrics Performance Leaderboard (Rank-ordered list of templates based on speed, cost, and alignment)",
            "Interactive Testing Sandbox (Variables filler panels with output generation simulators)",
            "System Cost & Token Run Graphs (Area chart representing token-in, token-out, and overall cost over time)"
        ],
        "test_cases": [
            "test_get_prompts_returns_all_templates",
            "test_register_new_version_increments_sequence",
            "test_trigger_evaluation_generates_correct_token_counts",
            "test_evaluation_saves_semantic_closeness_rating",
            "test_compare_endpoint_groups_by_template_name",
            "test_prompt_evaluation_handles_empty_fields",
            "test_compare_displays_percentage_efficiency_diffs",
            "test_health_endpoint_response_matches"
        ]
    },
    {
        "name": "trace-health-monitor",
        "title": "Distributed Tracing & Microservice Health Topology",
        "category": "DevOps & Cloud Infrastructure",
        "description": "Enterprise-grade service map tracker simulating transactional span logs, request tracing IDs, span duration bottlenecks, and DB latency trees.",
        "tech_stack": "FastAPI, Next.js 14, TailwindCSS, Docker Compose",
        "backend_endpoints": [
            {"method": "GET", "path": "/api/traces", "desc": "Retrieve trace logs and simulated spans"},
            {"method": "POST", "path": "/api/traces/span", "desc": "Ingest span event node representing sub-operation durations"},
            {"method": "GET", "path": "/api/topology/map", "desc": "Construct service dependency relational nodes and metrics"},
            {"method": "GET", "path": "/api/metrics/anomalies", "desc": "Identify trace execution nodes exceeding historical threshold standard deviations"},
            {"method": "GET", "path": "/api/health", "desc": "System self-health check"}
        ],
        "db_schema": [
            "trace_id: str",
            "span_id: str",
            "parent_span_id: str",
            "service_name: str",
            "operation_name: str",
            "duration_ms: float",
            "status_code: int"
        ],
        "sample_data": [
            {"trace_id": "tx-8802", "span_id": "sp-1", "parent_span_id": "", "service_name": "api-gateway", "operation_name": "POST /checkout", "duration_ms": 520.4, "status_code": 200},
            {"trace_id": "tx-8802", "span_id": "sp-2", "parent_span_id": "sp-1", "service_name": "auth-service", "operation_name": "JWT Decrypt", "duration_ms": 12.1, "status_code": 200},
            {"trace_id": "tx-8802", "span_id": "sp-3", "parent_span_id": "sp-1", "service_name": "payment-service", "operation_name": "Stripe API Charge", "duration_ms": 490.8, "status_code": 500}
        ],
        "frontend_sections": [
            "Interactive Service Topology Map (Visual network representing linked server components)",
            "Waterfall Trace Flamechart (Time-staggered operation bars highlighting duration bottlenecks)",
            "Trace Log Explorer (Filter lists searchable by Transaction ID, Service, or Exception flags)",
            "Anomaly Analysis Deck (Pulsing orange-red alerts flagging services with erratic standard deviations)"
        ],
        "test_cases": [
            "test_get_traces_returns_linked_spans",
            "test_ingest_span_registers_nested_node",
            "test_topology_endpoint_resolves_relations_map",
            "test_anomaly_detection_filters_high_durations",
            "test_broken_spans_return_failed_traces_correctly",
            "test_missing_parent_span_handled_as_root",
            "test_trace_id_search_returns_ordered_hierarchy",
            "test_health_check_operational"
        ]
    },
    {
        "name": "webhook-relay-engine",
        "title": "Secure Webhook Relay Gateway & Signature Auditing",
        "category": "Cybersecurity & Backend Middleware",
        "description": "High-security API relay executing automatic request routing, SHA-256 HMAC signature validations, endpoint retry plans, and payload inspection dashboards.",
        "tech_stack": "FastAPI, Next.js 14, TailwindCSS, Docker Compose",
        "backend_endpoints": [
            {"method": "GET", "path": "/api/endpoints", "desc": "List registered target downstream endpoints"},
            {"method": "POST", "path": "/api/endpoints", "desc": "Register new destination relay URL with custom secrets"},
            {"method": "POST", "path": "/api/relay/{endpoint_key}", "desc": "Ingest webhook, verify authorization signatures, and enqueue payload for retry delivery"},
            {"method": "GET", "path": "/api/relay/history", "desc": "Display relay audit logs with request/response headers"},
            {"method": "GET", "path": "/api/health", "desc": "System self-health check"}
        ],
        "db_schema": [
            "id: str (UUID)",
            "endpoint_key: str",
            "target_url: str",
            "hmac_secret: str",
            "delivery_attempts: int",
            "last_response_code: int",
            "status: str (active/failing)"
        ],
        "sample_data": [
            {"id": "e1", "endpoint_key": "stripe-receiver", "target_url": "https://api.mybiz.com/webhooks/payments", "hmac_secret": "whsec_abc123XYZ", "delivery_attempts": 1, "last_response_code": 200, "status": "active"},
            {"id": "e2", "endpoint_key": "slack-alert-relay", "target_url": "https://hooks.slack.com/services/T00/B00/X00", "hmac_secret": "whsec_slack_key", "delivery_attempts": 3, "last_response_code": 502, "status": "failing"}
        ],
        "frontend_sections": [
            "Downstream Registry Workspace (Config lists with endpoint key generators, URL targets, secret key reveal buttons)",
            "Live Webhook Relay Inspector (Double-panel payload viewer with source JSON and forwarded HTTP Headers)",
            "Delivery Attempts Chronology (Status timeline cards showing retry delays, timeouts, and error logs)",
            "Relay Metrics Cards (Counters for relays enqueued, signatures verified, failed, and success rate)"
        ],
        "test_cases": [
            "test_get_endpoints_returns_destination_configs",
            "test_register_endpoint_generates_hmac_secret",
            "test_relay_rejects_unsigned_webhooks_with_401",
            "test_relay_accepts_valid_signature_webhooks",
            "test_relay_updates_target_history_and_last_response",
            "test_failed_webhook_triggers_exponential_backoff",
            "test_endpoint_marked_failing_after_consecutive_errors",
            "test_health_returns_active_status"
        ]
    },
    {
        "name": "schema-drift-guard",
        "title": "DB Schema Drift Guard & Migration Engine",
        "category": "Enterprise SaaS Platform Tools",
        "description": "Continuous schema scanning dashboard that monitors target databases, identifies drift anomalies against base patterns, and builds migration scripts.",
        "tech_stack": "FastAPI, Next.js 14, TailwindCSS, Docker Compose",
        "backend_endpoints": [
            {"method": "GET", "path": "/api/databases", "desc": "List registered target database profiles"},
            {"method": "POST", "path": "/api/databases", "desc": "Add a new data store configuration for continuous auditing"},
            {"method": "GET", "path": "/api/scan/drift", "desc": "Compare active catalogs against reference templates to locate anomalies"},
            {"method": "POST", "path": "/api/scan/resolve", "desc": "Construct safe migration scripts resolving structural mismatches"},
            {"method": "GET", "path": "/api/health", "desc": "System self-health check"}
        ],
        "db_schema": [
            "id: str",
            "db_name: str",
            "connection_string: str",
            "sync_status: str (in-sync/drifted)",
            "last_scanned: datetime",
            "detected_anomalies: str (JSON text)"
        ],
        "sample_data": [
            {"id": "db1", "db_name": "Enterprise Production Postgres", "connection_string": "postgresql://usr:***@10.0.12.5:5432/main_prod", "sync_status": "drifted", "last_scanned": "2024-03-04T09:00:00Z", "detected_anomalies": '[{"type": "missing_column", "table": "users", "column": "mfa_secret", "expected": "VARCHAR(255)"}]'},
            {"id": "db2", "db_name": "Compliance Core SQL Server", "connection_string": "mssql://usr:***@10.0.12.9:1433/core_compliance", "sync_status": "in-sync", "last_scanned": "2024-03-04T09:15:00Z", "detected_anomalies": "[]"}
        ],
        "frontend_sections": [
            "Database Health Panel (A comprehensive listing of DB engines with status badges and sync metrics)",
            "Drift Delta Console (Side-by-side table comparisons indicating extra tables, missing columns, or type mismatches)",
            "Migration Script Editor (Auto-generated safe SQL scripts with copy buttons and custom risk indicators)",
            "Audit Schema History Ledger (Graph representing database alterations, migration application timestamps, and author IDs)"
        ],
        "test_cases": [
            "test_get_databases_returns_registered_stores",
            "test_add_database_persists_credentials_safely",
            "test_scan_drift_identifies_missing_table_entities",
            "test_scan_drift_flags_modified_column_types",
            "test_scan_drift_returns_empty_when_no_drift",
            "test_generate_migration_sql_includes_safe_guardrails",
            "test_apply_migration_resolves_drift_status_to_synced",
            "test_health_returns_guard_status"
        ]
    },
    {
        "name": "collab-canvas-engine",
        "title": "Real-Time Collaborative Canvas Sync Engine",
        "category": "Distributed Systems & Queues",
        "description": "Multiplayer whiteboard state engine simulating high-frequency websocket connection events, mouse cursor updates, and conflict-free replicated data types.",
        "tech_stack": "FastAPI, Next.js 14, TailwindCSS, Docker Compose",
        "backend_endpoints": [
            {"method": "GET", "path": "/api/canvases", "desc": "Retrieve lists of available canvas spaces and active guest counts"},
            {"method": "POST", "path": "/api/canvases", "desc": "Instantiate a new canvas whiteboard session space"},
            {"method": "GET", "path": "/api/canvases/{canvas_id}/state", "desc": "Query complete drawing shape inventory to seed joining browsers"},
            {"method": "POST", "path": "/api/canvases/{canvas_id}/sync", "desc": "Simulate ingestion of real-time coordinate streams, returning conflict resolutions"},
            {"method": "GET", "path": "/api/health", "desc": "System self-health check"}
        ],
        "db_schema": [
            "canvas_id: str",
            "title: str",
            "active_users: int",
            "shapes_count: int",
            "last_modified: datetime",
            "replicated_history: str (JSON shape inventory text)"
        ],
        "sample_data": [
            {"canvas_id": "c1", "title": "Quarterly Architecture Plan", "active_users": 14, "shapes_count": 89, "last_modified": "2024-03-05T16:00:00Z", "replicated_history": '[{"id": "rect-1", "type": "rectangle", "x": 120, "y": 250, "color": "#2563EB"}]'},
            {"canvas_id": "c2", "title": "Design System Ideas", "active_users": 3, "shapes_count": 12, "last_modified": "2024-03-05T16:42:00Z", "replicated_history": '[{"id": "circle-1", "type": "circle", "x": 500, "y": 450, "color": "#10B981"}]'}
        ],
        "frontend_sections": [
            "Interactive Draw Sandbox Simulator (Simulates live drawing components, adding boxes, drawing lines)",
            "Multiplayer Cursor Tracker Dashboard (Simulated active floating labels of collaborating team members)",
            "Drawing Element Stack Table (Lists all objects with undo, delete, and layering adjustments)",
            "Conflict Resolution Monitor Console (Live log tracing of coordinate merge events and CRDT sync counts)"
        ],
        "test_cases": [
            "test_get_canvases_returns_active_boards",
            "test_create_canvas_seeds_empty_history",
            "test_get_canvas_state_returns_valid_shapes_list",
            "test_sync_inserts_new_shape_to_history",
            "test_sync_resolves_timestamp_conflicts_using_crdt",
            "test_sync_updates_active_user_metrics",
            "test_delete_shapes_removes_from_replicated_history",
            "test_health_returns_canvas_engine_status"
        ]
    }
]


def ensure_history_dir():
    os.makedirs(os.path.dirname(PORTFOLIO_HISTORY_FILE), exist_ok=True)
    if not os.path.exists(PORTFOLIO_HISTORY_FILE):
        with open(PORTFOLIO_HISTORY_FILE, "w") as f:
            json.dump([], f)


def get_deployed_repos():
    ensure_history_dir()
    try:
        with open(PORTFOLIO_HISTORY_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []


def save_deployed_repo(repo_meta):
    ensure_history_dir()
    repos = get_deployed_repos()
    repos.append(repo_meta)
    with open(PORTFOLIO_HISTORY_FILE, "w") as f:
        json.dump(repos, f, indent=2)


def select_next_blueprint():
    deployed = get_deployed_repos()
    deployed_names = {r["name"] for r in deployed}
    print(f"History ledger lists {len(deployed_names)} built repos: {deployed_names}")
    for bp in BLUEPRINTS:
        if bp["name"] not in deployed_names:
            return bp
    print("Cycle completed! Auto-selecting first blueprint to overwrite/upgrade.")
    return BLUEPRINTS[0]


def generate_with_groq(system_prompt, user_prompt, max_tokens=950):
    max_tokens = min(max_tokens, 950)
    user_prompt = user_prompt.strip()
    system_prompt = system_prompt.strip()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    attempts = 5
    delay = 35.0
    for attempt in range(attempts):
        try:
            response = litellm.completion(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            err_str = str(e)
            print(f"  [LLM Error] {err_str[:200]}")
            if "RateLimitError" in err_str or "429" in err_str:
                print(f"  [Rate Limit Hit] Sleeping {delay}s (attempt {attempt+1}/{attempts})...")
                time.sleep(delay)
                delay *= 2.0
            else:
                raise e
    raise RuntimeError(f"Failed to generate code after {attempts} attempts.")


def generate_backend_code(blueprint):
    print("\n--- [Stage 1/3] Generating Production FastAPI Backend (main.py) ---")
    sys_prompt = (
        "You are a Senior Staff Software Engineer specialized in high-performance FastAPI backends.\n"
        "Write a single, production-grade, complete main.py file.\n"
        "STRICT CONSTRAINTS:\n"
        "- Return ONLY the raw Python code inside a python markdown block.\n"
        "- No descriptions, notes, or explanations outside the code block.\n"
        "- Use Pydantic v2 conventions.\n"
        "- Include CORS middleware with wildcard origins.\n"
        "- Implement EVERY endpoint from the blueprint. No placeholders or TODOs.\n"
        "- Use docstrings, typing, explicit status codes, error handlers (404/400).\n"
        "- Pre-populate in-memory DB with exact mock records from blueprint.\n"
        "- Keep code dense and compact. Avoid verbose comments."
    )
    user_prompt = (
        f"BLUEPRINT: {blueprint['name']}\n"
        f"TITLE: {blueprint['title']}\n"
        f"DB SCHEMA:\n{json.dumps(blueprint['db_schema'], indent=2)}\n"
        f"ENDPOINTS:\n{json.dumps(blueprint['backend_endpoints'], indent=2)}\n"
        f"SAMPLE DATA:\n{json.dumps(blueprint['sample_data'], indent=2)}\n"
        "Write the complete main.py. All POST endpoints must mutate the in-memory dataset."
    )
    return generate_with_groq(sys_prompt, user_prompt, max_tokens=950)


def generate_test_code(blueprint, backend_code):
    print("\n--- [Stage 2/3] Generating Pytest Test Suite (test_main.py) ---")
    sys_prompt = (
        "You are an Elite QA Automation Architect.\n"
        "Write a complete, executable pytest suite for the provided FastAPI backend.\n"
        "STRICT CONSTRAINTS:\n"
        "- Return ONLY the raw Python code inside a python markdown block.\n"
        "- No descriptions or footnotes.\n"
        "- Implement EXACTLY 8 tests matching the blueprint test case names.\n"
        "- Use fastapi.testclient.TestClient and standard pytest assertions.\n"
        "- Write clean, robust tests calling actual backend endpoints.\n"
        "- Keep code compact and free of verbose comments."
    )
    user_prompt = (
        f"BLUEPRINT: {blueprint['name']}\n"
        f"TEST CASES (EXACTLY THESE 8):\n{json.dumps(blueprint['test_cases'], indent=2)}\n"
        f"TARGET CODE:\n{B3}python\n{backend_code}\n{B3}\n"
        "Write the complete test_main.py."
    )
    return generate_with_groq(sys_prompt, user_prompt, max_tokens=950)


def generate_page_code(blueprint):
    print("\n--- [Stage 3/3] Generating Responsive Next.js 14 UI (page.tsx) ---")
    sys_prompt = (
        "You are a World-Class Frontend UI Engineer specialized in Next.js 14 App Router and TailwindCSS.\n"
        "Write a single, production-ready React component file page.tsx.\n"
        "STRICT CONSTRAINTS:\n"
        "- Return ONLY the raw TSX code inside a typescript markdown block.\n"
        "- No descriptions or notes outside the code block.\n"
        "- Must start with 'use client';\n"
        "- Connect to backend: const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';\n"
        "- Use TailwindCSS for professional dark-themed responsive UI.\n"
        "- Implement loading states, empty states, and form validation.\n"
        "- Implement EVERY section from the blueprint. No mocks or placeholders.\n"
        "- Use raw inline SVG elements for icons, NOT Lucide or external icon libraries.\n"
        "- Keep code dense, compact, and completely self-contained."
    )
    user_prompt = (
        f"BLUEPRINT: {blueprint['name']}\n"
        f"TITLE: {blueprint['title']}\n"
        f"DESCRIPTION: {blueprint['description']}\n"
        f"PAGE SECTIONS:\n{json.dumps(blueprint['frontend_sections'], indent=2)}\n"
        f"API ENDPOINTS:\n{json.dumps(blueprint['backend_endpoints'], indent=2)}\n"
        "Write the complete page.tsx with robust state-management handlers."
    )
    return generate_with_groq(sys_prompt, user_prompt, max_tokens=950)


def parse_code_block(raw_text, language="python"):
    raw_text = raw_text.strip()
    patterns = [
        rf"^{B3}{language}\s*\n(.*?)\n{B3}$",
        rf"^{B3}\s*\n(.*?)\n{B3}$",
        r"^(.*?)$"
    ]
    for pattern in patterns:
        match = re.match(pattern, raw_text, re.DOTALL | re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
            if extracted:
                return extracted
    return raw_text


def run_layer1_review(file_path, code_text):
    print(f"  Layer 1 (Syntax): Scanning '{file_path}'...")
    sys_prompt = (
        "You are an automated code syntax analyzer.\n"
        "Check for syntax errors, incomplete truncation, unclosed brackets, quotes.\n"
        "If perfect, return ONLY 'ALL_PASS'.\n"
        "If errors found, return corrected code inside markdown blocks. No descriptions."
    )
    user_prompt = f"FILE: {file_path}\nCODE:\n{code_text}"
    review_out = generate_with_groq(sys_prompt, user_prompt, max_tokens=950)
    if "ALL_PASS" in review_out:
        return "ALL_PASS", code_text
    lang = "python" if file_path.endswith(".py") else "typescript"
    corrected = parse_code_block(review_out, lang)
    if len(corrected) < len(code_text) * 0.70:
        print(f"  Warning: Layer 1 returned truncated code. Keeping original.")
        return "ALL_PASS", code_text
    return "CORRECTED", corrected


def run_layer2_review(backend_code, frontend_code):
    print(f"  Layer 2 (Integration): Validating API connections...")
    sys_prompt = (
        "You are a full-stack integration inspector.\n"
        "Verify Next.js page.tsx fetch calls match FastAPI routes in main.py.\n"
        "Check endpoint paths, HTTP methods, parameter models, API_BASE usage.\n"
        "If aligned, return 'ALL_PASS'.\n"
        "If bugs found, output corrected page.tsx inside typescript markdown blocks."
    )
    user_prompt = f"BACKEND:\n{backend_code}\n\nFRONTEND:\n{frontend_code}"
    review_out = generate_with_groq(sys_prompt, user_prompt, max_tokens=950)
    if "ALL_PASS" in review_out:
        return "ALL_PASS", frontend_code
    corrected = parse_code_block(review_out, "typescript")
    if len(corrected) < len(frontend_code) * 0.70:
        print(f"  Warning: Layer 2 returned truncated code. Keeping original.")
        return "ALL_PASS", frontend_code
    return "CORRECTED", corrected


def run_layer3_review(file_path, code_text, blueprint):
    print(f"  Layer 3 (Completeness): Checking '{file_path}'...")
    sys_prompt = (
        "You are a QA Auditor inspecting feature specifications.\n"
        "Compare code with blueprint. Check endpoints, tests, sections are complete.\n"
        "Ensure no TODO comments, dummy mocks, or empty controllers.\n"
        "If complete, return ONLY 'ALL_PASS'.\n"
        "If omissions found, return fully expanded file inside markdown blocks."
    )
    user_prompt = f"BLUEPRINT:\n{json.dumps(blueprint, indent=2)}\n\nFILE: {file_path}\nCODE:\n{code_text}"
    review_out = generate_with_groq(sys_prompt, user_prompt, max_tokens=950)
    if "ALL_PASS" in review_out:
        return "ALL_PASS", code_text
    lang = "python" if file_path.endswith(".py") else "typescript"
    corrected = parse_code_block(review_out, lang)
    if len(corrected) < len(code_text) * 0.70:
        print(f"  Warning: Layer 3 returned truncated code. Keeping original.")
        return "ALL_PASS", code_text
    return "CORRECTED", corrected


def run_deterministic_checks(backend, test_code, frontend):
    print("Running Phase 3 (Deterministic Checks)...")
    issues = []
    try:
        compile(backend, "main.py", "exec")
        print("  Backend compiles OK.")
    except SyntaxError as e:
        issues.append(f"Backend syntax error line {e.lineno}: {e.msg}")
    try:
        compile(test_code, "test_main.py", "exec")
        print("  Test suite compiles OK.")
    except SyntaxError as e:
        issues.append(f"Test syntax error line {e.lineno}: {e.msg}")
    if "use client" not in frontend:
        issues.append("Frontend missing 'use client' directive.")
    if "fetch(" not in frontend:
        issues.append("Frontend does not use fetch().")
    if "API_BASE" not in frontend:
        issues.append("Frontend missing API_BASE variable.")
    for path, code in [("main.py", backend), ("test_main.py", test_code), ("page.tsx", frontend)]:
        lines = [l.strip() for l in code.split("\n") if l.strip()]
        if lines:
            last = lines[-1]
            if last.endswith(("=", "+", ",", "or", "and", "?", ":", ".")):
                issues.append(f"Truncation: {path} ends with '{last}'")
    if issues:
        print(f"  Found {len(issues)} issues:")
        for i in issues:
            print(f"    - {i}")
        return False, issues
    print("  All deterministic checks passed!")
    return True, []


def emergency_correct_code(path, code, issues):
    print(f"Emergency Patch for '{path}'...")
    sys_prompt = (
        "You are an Emergency Hotfix Developer.\n"
        "Fix specific issues in the provided source file.\n"
        "Return ONLY corrected code inside markdown blocks.\n"
        "Preserve entire structure. Fix only identified gaps.\n"
        "Ensure no unclosed strings, brackets, or blocks."
    )
    user_prompt = f"FILE: {path}\nISSUES:\n" + "\n".join(issues) + f"\n\nCODE:\n{code}"
    out = generate_with_groq(sys_prompt, user_prompt, max_tokens=950)
    lang = "python" if path.endswith(".py") else "typescript"
    corrected = parse_code_block(out, lang)
    if len(corrected) > len(code) * 0.70:
        return corrected
    return code


def publish_repository(blueprint, backend, test_code, frontend, run_meta):
    print("\n=================================================================")
    print("  PHASE 5: REPOSITORY DEPLOYMENT")
    print("=================================================================")
    pat = os.environ.get("GITHUB_PAT") or os.environ.get("GITHUB_TOKEN")
    if not pat:
        print("Error: GITHUB_PAT or GITHUB_TOKEN missing. Skip deployment.")
        return False
    owner = "patel-tirth-1952008"
    repo_name = blueprint["name"]
    headers = {"Authorization": f"token {pat}", "Accept": "application/vnd.github.v3+json"}

    print(f"Creating GitHub repository '{owner}/{repo_name}'...")
    create_payload = {
        "name": repo_name,
        "description": f"{blueprint['title']} - Generated by AI Agent 1 v2",
        "private": False,
        "has_issues": True,
        "has_projects": False,
        "has_wiki": False
    }
    res = requests.post("https://api.github.com/user/repos", headers=headers, json=create_payload)
    if res.status_code == 201:
        print("  Repository created!")
    elif res.status_code == 422:
        print("  Repository already exists. Updating files...")
    else:
        print(f"  Failed to create repo: {res.status_code} - {res.text}")
        return False

    year = datetime.now().year
    api_table_rows = "\n".join(
        [f"| `{ep['method']}` | `{ep['path']}` | {ep['desc']} |" for ep in blueprint['backend_endpoints']]
    )

    readme_content = (
        f"# {blueprint['title']}\n\n"
        f"[![CI](https://github.com/{owner}/{repo_name}/actions/workflows/ci.yml/badge.svg)]"
        f"(https://github.com/{owner}/{repo_name}/actions)\n"
        f"[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]"
        f"(https://opensource.org/licenses/MIT)\n"
        f"[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg?style=flat&logo=fastapi)]"
        f"(https://fastapi.tiangolo.com)\n"
        f"[![Next.js 14](https://img.shields.io/badge/Frontend-Next.js%2014-000000.svg?style=flat&logo=nextdotjs)]"
        f"(https://nextjs.org)\n\n"
        f"{blueprint['description']}\n\n"
        f"## Architecture\n"
        f"{B3}\n"
        f"Next.js 14 Frontend (Port 3000)\n"
        f"        |\n"
        f"   fetch() API\n"
        f"        v\n"
        f"FastAPI Backend (Port 8000)\n"
        f"{B3}\n\n"
        f"## API Table\n\n"
        f"| Method | Endpoint | Description |\n"
        f"|:---|:---|:---|\n"
        f"{api_table_rows}\n\n"
        f"## Quickstart\n\n"
        f"{B3}bash\n"
        f"docker compose up --build\n"
        f"{B3}\n\n"
        f"- Frontend: [http://localhost:3000](http://localhost:3000)\n"
        f"- Backend: [http://localhost:8000](http://localhost:8000)\n"
        f"- API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)\n\n"
        f"## Tests\n\n"
        f"{B3}bash\n"
        f"cd backend && pip install -r requirements.txt && pytest\n"
        f"{B3}\n\n"
        f"Generated by **AI Agent 1 v2** with 3-layer self-correction review pipeline.\n"
    )

    ci_yml = (
        "name: CI\n\n"
        "on:\n"
        "  push:\n"
        "    branches: [ main ]\n"
        "  pull_request:\n"
        "    branches: [ main ]\n\n"
        "jobs:\n"
        "  test-backend:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/checkout@v4\n"
        "      - uses: actions/setup-python@v5\n"
        "        with:\n"
        "          python-version: '3.12'\n"
        "      - run: |\n"
        "          cd backend\n"
        "          pip install -r requirements.txt\n"
        "          pytest\n\n"
        "  test-frontend:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/checkout@v4\n"
        "      - uses: actions/setup-node@v4\n"
        "        with:\n"
        "          node-version: '20'\n"
        "      - run: |\n"
        "          cd frontend\n"
        "          npm install\n"
        "          npm run build\n"
    )

    files_to_deploy = {
        "LICENSE": (
            f"MIT License\n\nCopyright (c) {year} {owner}\n\n"
            "Permission is hereby granted, free of charge, to any person obtaining a copy\n"
            "of this software and associated documentation files (the \"Software\"), to deal\n"
            "in the Software without restriction, including without limitation the rights\n"
            "to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n"
            "copies of the Software, and to permit persons to whom the Software is\n"
            "furnished to do so, subject to the following conditions:\n\n"
            "The above copyright notice and this permission notice shall be included in all\n"
            "copies or substantial portions of the Software.\n\n"
            "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n"
            "IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n"
            "FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n"
            "AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n"
            "LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n"
            "OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\n"
            "SOFTWARE.\n"
        ),
        ".env.example": "PORT=8000\nNEXT_PUBLIC_API_URL=http://localhost:8000\n",
        ".gitignore": "__pycache__/\n*.py[cod]\n.pytest_cache/\n.env\nnode_modules/\n.next/\nout/\n.DS_Store\n",
        "README.md": readme_content,
        "docker-compose.yml": (
            "services:\n"
            "  backend:\n"
            "    build: ./backend\n"
            "    ports:\n"
            '      - "8000:8000"\n'
            "    environment:\n"
            "      - PORT=8000\n"
            "    volumes:\n"
            "      - ./backend:/app\n\n"
            "  frontend:\n"
            "    build: ./frontend\n"
            "    ports:\n"
            '      - "3000:3000"\n'
            "    environment:\n"
            "      - NEXT_PUBLIC_API_URL=http://localhost:8000\n"
            "    depends_on:\n"
            "      - backend\n"
        ),
        ".github/workflows/ci.yml": ci_yml,
        "backend/requirements.txt": (
            "fastapi==0.110.0\n"
            "uvicorn[standard]==0.27.1\n"
            "pydantic==2.6.4\n"
            "pytest==8.0.2\n"
            "httpx==0.27.0\n"
            "python-multipart==0.0.9\n"
        ),
        "backend/Dockerfile": (
            "FROM python:3.12-slim\n"
            "WORKDIR /app\n"
            "COPY requirements.txt .\n"
            "RUN pip install --no-cache-dir -r requirements.txt\n"
            "COPY . .\n"
            "EXPOSE 8000\n"
            'CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]\n'
        ),
        "backend/main.py": backend,
        "backend/test_main.py": test_code,
        "frontend/Dockerfile": (
            "FROM node:20-slim\n"
            "WORKDIR /app\n"
            "COPY package*.json ./\n"
            "RUN npm install\n"
            "COPY . .\n"
            "RUN npm run build\n"
            "EXPOSE 3000\n"
            'CMD ["npm", "run", "dev"]\n'
        ),
        "frontend/package.json": (
            '{\n'
            '  "name": "portfolio-frontend",\n'
            '  "version": "0.1.0",\n'
            '  "private": true,\n'
            '  "scripts": {\n'
            '    "dev": "next dev -H 0.0.0.0 -p 3000",\n'
            '    "build": "next build",\n'
            '    "start": "next start",\n'
            '    "lint": "next lint"\n'
            '  },\n'
            '  "dependencies": {\n'
            '    "next": "14.1.0",\n'
            '    "react": "^18.2.0",\n'
            '    "react-dom": "^18.2.0"\n'
            '  },\n'
            '  "devDependencies": {\n'
            '    "typescript": "^5.3.3",\n'
            '    "@types/node": "^20.11.24",\n'
            '    "@types/react": "^18.2.61",\n'
            '    "@types/react-dom": "^18.2.19",\n'
            '    "autoprefixer": "^10.4.18",\n'
            '    "postcss": "^8.4.35",\n'
            '    "tailwindcss": "^3.4.1"\n'
            '  }\n'
            '}\n'
        ),
        "frontend/tsconfig.json": (
            '{\n'
            '  "compilerOptions": {\n'
            '    "target": "es5",\n'
            '    "lib": ["dom", "dom.iterable", "esnext"],\n'
            '    "allowJs": true,\n'
            '    "skipLibCheck": true,\n'
            '    "strict": true,\n'
            '    "noEmit": true,\n'
            '    "esModuleInterop": true,\n'
            '    "module": "esnext",\n'
            '    "moduleResolution": "bundler",\n'
            '    "resolveJsonModule": true,\n'
            '    "isolatedModules": true,\n'
            '    "jsx": "preserve",\n'
            '    "incremental": true,\n'
            '    "plugins": [{"name": "next"}],\n'
            '    "paths": {"@/*": ["./src/*"]}\n'
            '  },\n'
            '  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],\n'
            '  "exclude": ["node_modules"]\n'
            '}\n'
        ),
        "frontend/next.config.js": (
            "/** @type {import('next').NextConfig} */\n"
            "const nextConfig = {\n"
            "  reactStrictMode: true,\n"
            "  typescript: { ignoreBuildErrors: true },\n"
            "  eslint: { ignoreDuringBuilds: true }\n"
            "};\n"
            "module.exports = nextConfig;\n"
        ),
        "frontend/tailwind.config.js": (
            "/** @type {import('tailwindcss').Config} */\n"
            "module.exports = {\n"
            '  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],\n'
            "  theme: { extend: {} },\n"
            "  plugins: [],\n"
            "};\n"
        ),
        "frontend/postcss.config.js": (
            "module.exports = {\n"
            "  plugins: { tailwindcss: {}, autoprefixer: {} },\n"
            "};\n"
        ),
        "frontend/src/app/globals.css": (
            "@tailwind base;\n"
            "@tailwind components;\n"
            "@tailwind utilities;\n\n"
            "body {\n"
            "  color: #f3f4f6;\n"
            "  background-color: #0b0f19;\n"
            "}\n"
        ),
        "frontend/src/app/layout.tsx": (
            "import './globals.css';\n"
            "import type { Metadata } from 'next';\n\n"
            "export const metadata: Metadata = {\n"
            f"  title: '{blueprint['title']}',\n"
            f"  description: '{blueprint['description']}',\n"
            "};\n\n"
            "export default function RootLayout({ children }: { children: React.ReactNode }) {\n"
            "  return (\n"
            "    <html lang=\"en\">\n"
            "      <body>{children}</body>\n"
            "    </html>\n"
            "  );\n"
            "}\n"
        ),
        "frontend/src/app/page.tsx": frontend
    }

    for file_path, content in files_to_deploy.items():
        print(f"  Deploying '{file_path}'...")
        url = f"https://api.github.com/repos/{owner}/{repo_name}/contents/{file_path}"
        sha = None
        check_res = requests.get(url, headers=headers)
        if check_res.status_code == 200:
            sha = check_res.json().get("sha")
        payload = {
            "message": f"autobot: deploy {file_path}",
            "content": base64.b64encode(content.encode("utf-8")).decode("utf-8")
        }
        if sha:
            payload["sha"] = sha
        put_res = requests.put(url, headers=headers, json=payload)
        if put_res.status_code not in [200, 201]:
            print(f"    Error uploading {file_path}: {put_res.status_code}")
            return False

    print(f"Repository published: https://github.com/{owner}/{repo_name}")
    return True


def run_portfolio_builder():
    print("=================================================================")
    print("  AGENT 1: PORTFOLIO BUILDER v2 (Multi-Layer Review)")
    print(f"  Model: {MODEL_NAME} | Review Cycles: {MAX_REVIEW_CYCLES}")
    print("=================================================================")

    blueprint = select_next_blueprint()
    print(f"\n[Selected Blueprint] {blueprint['title']}")
    print(f"Category:    {blueprint['category']}")
    print(f"Repository:  {blueprint['name']}")
    print(f"Description: {blueprint['description']}")

    # PHASE 1: GENERATION
    print("\n=================================================================")
    print("  PHASE 1: CODE GENERATION")
    print("=================================================================")

    backend_raw = generate_backend_code(blueprint)
    backend_code = parse_code_block(backend_raw, "python")
    print("  Backend generation complete.")
    print(f"Waiting {STAGE_DELAY}s for rate-limit reset...")
    time.sleep(STAGE_DELAY)

    test_raw = generate_test_code(blueprint, backend_code)
    test_code = parse_code_block(test_raw, "python")
    print("  Test suite generation complete.")
    print(f"Waiting {STAGE_DELAY}s for rate-limit reset...")
    time.sleep(STAGE_DELAY)

    frontend_raw = generate_page_code(blueprint)
    frontend_code = parse_code_block(frontend_raw, "typescript")
    print("  Frontend generation complete.")

    # PHASE 2: MULTI-LAYER REVIEW
    print("\n=================================================================")
    print("  PHASE 2: 3-LAYER SELF-CORRECTION REVIEW")
    print("=================================================================")

    review_success = True
    for cycle in range(1, MAX_REVIEW_CYCLES + 1):
        print(f"\n--- [Review Cycle {cycle}/{MAX_REVIEW_CYCLES}] ---")
        cycle_corrections = 0
        print(f"Waiting {REVIEW_DELAY}s to clear rate limits...")
        time.sleep(REVIEW_DELAY)

        l1s, backend_code = run_layer1_review("backend/main.py", backend_code)
        if l1s == "CORRECTED":
            cycle_corrections += 1
            time.sleep(REVIEW_DELAY)

        l1t, test_code = run_layer1_review("backend/test_main.py", test_code)
        if l1t == "CORRECTED":
            cycle_corrections += 1
            time.sleep(REVIEW_DELAY)

        l1f, frontend_code = run_layer1_review("frontend/src/app/page.tsx", frontend_code)
        if l1f == "CORRECTED":
            cycle_corrections += 1
            time.sleep(REVIEW_DELAY)

        l2s, frontend_code = run_layer2_review(backend_code, frontend_code)
        if l2s == "CORRECTED":
            cycle_corrections += 1
            time.sleep(REVIEW_DELAY)

        l3b, backend_code = run_layer3_review("backend/main.py", backend_code, blueprint)
        if l3b == "CORRECTED":
            cycle_corrections += 1
            time.sleep(REVIEW_DELAY)

        l3f, frontend_code = run_layer3_review("frontend/src/app/page.tsx", frontend_code, blueprint)
        if l3f == "CORRECTED":
            cycle_corrections += 1

        print(f"Cycle {cycle} done. Applied {cycle_corrections} corrections.")
        if cycle_corrections == 0:
            print("Codebase passed all review layers!")
            break
    else:
        print("Reached max review cycles. Proceeding.")
        review_success = False

    # PHASE 3 & 4: DETERMINISTIC + EMERGENCY
    passed_checks, issues = run_deterministic_checks(backend_code, test_code, frontend_code)

    if not passed_checks:
        print("\n=================================================================")
        print("  PHASE 4: EMERGENCY MITIGATION")
        print("=================================================================")
        backend_issues = [i for i in issues if "backend" in i.lower() or "main.py" in i]
        if backend_issues:
            backend_code = emergency_correct_code("backend/main.py", backend_code, backend_issues)
            time.sleep(REVIEW_DELAY)
        test_issues = [i for i in issues if "test" in i.lower()]
        if test_issues:
            test_code = emergency_correct_code("backend/test_main.py", test_code, test_issues)
            time.sleep(REVIEW_DELAY)
        front_issues = [i for i in issues if "front" in i.lower() or "page.tsx" in i or "truncation" in i.lower()]
        if front_issues:
            frontend_code = emergency_correct_code("frontend/src/app/page.tsx", frontend_code, front_issues)
        passed_checks, issues = run_deterministic_checks(backend_code, test_code, frontend_code)

    run_meta = {
        "name": blueprint["name"],
        "title": blueprint["title"],
        "timestamp": datetime.now().isoformat(),
        "review_passed": review_success and passed_checks,
        "metrics": {"issues_found": len(issues), "final_checks_passed": passed_checks}
    }

    success = publish_repository(blueprint, backend_code, test_code, frontend_code, run_meta)

    if success:
        save_deployed_repo(run_meta)
        print("\nPortfolio Build completed! Deployment pushed to GitHub.")
        try:
            print("Saving history to main repository...")
            subprocess.run(["git", "config", "global", "user.name", "github-actions[bot]"], check=True)
            subprocess.run(["git", "config", "global", "user.email", "github-actions[bot]@users.noreply.github.com"], check=True)
            subprocess.run(["git", "add", "data/portfolio_history.json"], check=True)
            subprocess.run(["git", "stash"], check=False)
            subprocess.run(["git", "pull", "--rebase", "origin", "main"], check=True)
            subprocess.run(["git", "stash", "pop"], check=False)
            subprocess.run(["git", "commit", "-m", f"chore: update history for {blueprint['name']}"], check=True)
            subprocess.run(["git", "push", "origin", "main"], check=True)
            print("  History pushed to control repository.")
        except Exception as git_err:
            print(f"  Warning: Failed to push history: {git_err}")
    else:
        print("\nPortfolio Builder exited: Deployment failed.")


if __name__ == "__main__":
    run_portfolio_builder()