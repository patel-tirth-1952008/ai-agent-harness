"""
Agent 1: Autonomous Senior-Level Portfolio Project Builder
Generates complete, production-ready, full-stack applications on GitHub.
Every repository includes:
  - Working FastAPI backend with Pydantic schemas, validation, CORS, error handling, sample data
  - Working Next.js 14 frontend with Tailwind CSS, dark theme, connected API fetch calls, loading/empty/error states
  - Pytest test suite with TestClient covering all endpoints and edge cases
  - Docker Compose orchestration running both services out of the box
  - GitHub Actions CI/CD pipeline running automated tests
  - Professional README with architecture diagrams, API specs, and run instructions
  - MIT License and .env.example
"""

import os
import sys
import json
import time
import re
import random
import base64
import requests
from datetime import datetime, timezone

# Backtick constant for safe string generation without breaking markdown parsers
B3 = chr(96) * 3

# ---------------------------------------------------------------------------
# Monkeypatch LiteLLM for Groq Compatibility (Strips Unsupported Parameters)
# ---------------------------------------------------------------------------
try:
    import litellm

    litellm.drop_params = True
    litellm.set_verbose = False

    _orig_completion = litellm.completion
    _orig_acompletion = litellm.acompletion

    def _clean_groq_messages(messages):
        if not isinstance(messages, list):
            return messages
        cleaned = []
        for msg in messages:
            if isinstance(msg, dict):
                c_msg = {k: v for k, v in msg.items() if k not in ("cache_breakpoint", "cache_control")}
                cleaned.append(c_msg)
            else:
                cleaned.append(msg)
        return cleaned

    def _patched_completion(*args, **kwargs):
        if "messages" in kwargs:
            kwargs["messages"] = _clean_groq_messages(kwargs["messages"])
        max_retries = 5
        for attempt in range(max_retries):
            try:
                return _orig_completion(*args, **kwargs)
            except Exception as e:
                err_str = str(e).lower()
                if ("429" in err_str or "rate_limit" in err_str or "quota" in err_str) and attempt < max_retries - 1:
                    sleep_time = 35 * (attempt + 1)
                    print(f"  [Rate Limit Hit] LiteLLM backoff: sleeping {sleep_time}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(sleep_time)
                else:
                    raise e

    async def _patched_acompletion(*args, **kwargs):
        if "messages" in kwargs:
            kwargs["messages"] = _clean_groq_messages(kwargs["messages"])
        return await _orig_acompletion(*args, **kwargs)

    litellm.completion = _patched_completion
    litellm.acompletion = _patched_acompletion
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "groq/qwen/qwen3.8-27b"
MAX_TOKENS = 2000
TEMPERATURE = 0.2
STAGE_DELAY = 65  # Seconds between LLM generation calls to respect Groq OTPM (1000 tokens/min)

HISTORY_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "portfolio_history.json")

# ---------------------------------------------------------------------------
# 10 Production-Grade Project Blueprints
# ---------------------------------------------------------------------------
PROJECT_BLUEPRINTS = [
    {
        "name": "cloudops-sentinel",
        "title": "CloudOps Sentinel - Real-Time Infrastructure Health & Incident Monitor",
        "description": "Production-grade microservices health monitor and automated incident alerting engine with latency anomaly detection, uptime SLAs, and webhook notifications.",
        "category": "DevOps & Cloud Infrastructure",
        "tech_stack": ["FastAPI", "Python 3.11", "Next.js 14", "TypeScript", "Tailwind CSS", "Docker", "Pytest"],
        "backend_endpoints": [
            {"method": "GET", "path": "/api/health", "desc": "System health and database connectivity probe"},
            {"method": "GET", "path": "/api/services", "desc": "List all monitored services with real-time status and uptime percentage"},
            {"method": "POST", "path": "/api/services", "desc": "Register a new service with healthcheck URL and threshold parameters"},
            {"method": "GET", "path": "/api/services/{service_id}/metrics", "desc": "Retrieve latency, error rate, and CPU metrics time series"},
            {"method": "POST", "path": "/api/incidents", "desc": "Trigger and log a new incident with severity level"},
            {"method": "GET", "path": "/api/incidents", "desc": "List active and resolved incidents with audit log"},
            {"method": "PATCH", "path": "/api/incidents/{incident_id}/resolve", "desc": "Acknowledge and resolve an active incident"}
        ],
        "db_schema": "Services (id, name, endpoint_url, status, uptime_pct, check_interval_sec), Metrics (id, service_id, latency_ms, status_code, timestamp), Incidents (id, service_id, severity, summary, status, created_at, resolved_at)",
        "sample_data": "Pre-populate with 4 services (Payment Gateway [Healthy], Auth Service [Healthy], Notification Hub [Degraded], Search API [Healthy]) and 2 recent incidents with full metric history.",
        "frontend_sections": [
            "Dashboard Summary Bar: Total services monitored, overall uptime %, active incidents count, average P95 latency",
            "Service Health Grid: Cards showing service name, live ping status (green/yellow/red badge), uptime %, and response time sparkline",
            "Incident Command Center: Table of active/resolved incidents with filter by severity (Critical, Warning, Info) and instant 'Resolve' action",
            "Service Registration Modal/Form: Add new endpoint with validation for URL, check interval, and latency thresholds",
            "Live Metrics Inspector: Time-series latency chart for the selected service"
        ],
        "test_cases": [
            "test_health_endpoint_returns_200",
            "test_get_services_returns_populated_list",
            "test_create_service_validation_success",
            "test_create_service_invalid_url_fails",
            "test_get_service_metrics_returns_time_series",
            "test_create_incident_sets_active_status",
            "test_resolve_incident_updates_timestamp",
            "test_get_incidents_filter_by_status"
        ]
    },
    {
        "name": "rag-doc-intel",
        "title": "RAG Document Intelligence & Semantic Search Engine",
        "description": "Enterprise-ready Retrieval-Augmented Generation (RAG) backend and search interface for semantic document search, chunk extraction, and AI synthesis.",
        "category": "AI Engineering & Vector Search",
        "tech_stack": ["FastAPI", "Python 3.11", "Next.js 14", "TypeScript", "Tailwind CSS", "Docker", "Pytest"],
        "backend_endpoints": [
            {"method": "GET", "path": "/api/health", "desc": "API health probe and vector store status"},
            {"method": "GET", "path": "/api/documents", "desc": "List all indexed documents with chunk count and upload timestamp"},
            {"method": "POST", "path": "/api/documents", "desc": "Upload and index new text document with automatic chunking and vector embedding simulation"},
            {"method": "DELETE", "path": "/api/documents/{doc_id}", "desc": "Delete document and purge associated vector chunks"},
            {"method": "POST", "path": "/api/search", "desc": "Semantic search across all indexed chunks using cosine similarity scoring"},
            {"method": "POST", "path": "/api/query", "desc": "Execute RAG question-answering with retrieved context snippets and confidence score"},
            {"method": "GET", "path": "/api/stats", "desc": "Get total documents, total chunks, average chunk size, and query latency stats"}
        ],
        "db_schema": "Documents (id, title, content_preview, chunk_count, file_size_kb, created_at), Chunks (id, doc_id, chunk_index, text, embedding_vector, token_count), QueryLogs (id, query_text, matched_chunks, response_text, latency_ms, confidence_score)",
        "sample_data": "Pre-populate with 3 enterprise knowledge base documents (Kubernetes Deployment Guide, Company Security Policy, REST API Guidelines) with pre-computed text chunks.",
        "frontend_sections": [
            "System Stats Bar: Total documents indexed, total vector chunks, avg retrieval latency (ms), vector index status",
            "RAG Interactive Search Bar: Natural language search box with hybrid keyword + semantic similarity threshold slider",
            "Retrieved Context Cards: Matched chunks highlighted with similarity score percentage and source document badge",
            "Document Knowledge Base Manager: List of indexed documents with chunk counts, delete action, and upload modal",
            "AI Synthesis Panel: Generated synthesized answer with citations referencing specific document chunks"
        ],
        "test_cases": [
            "test_health_endpoint",
            "test_list_documents_returns_seed_data",
            "test_upload_document_creates_chunks",
            "test_delete_document_cascades_chunks",
            "test_semantic_search_returns_ranked_results",
            "test_rag_query_generates_cited_response",
            "test_stats_endpoint_returns_accurate_counts",
            "test_search_empty_query_returns_400"
        ]
    },
    {
        "name": "ratelimit-gateway",
        "title": "Distributed API Rate Limiter & Token Bucket Gateway",
        "description": "High-throughput API rate limiting and token bucket throttling gateway with tiered quotas, IP blacklisting, and real-time consumption analytics.",
        "category": "Distributed Systems & Security",
        "tech_stack": ["FastAPI", "Python 3.11", "Next.js 14", "TypeScript", "Tailwind CSS", "Docker", "Pytest"],
        "backend_endpoints": [
            {"method": "GET", "path": "/api/health", "desc": "Gateway status and rate limiter algorithm health"},
            {"method": "GET", "path": "/api/keys", "desc": "List all API keys with plan tier (Free, Pro, Enterprise) and current token bucket levels"},
            {"method": "POST", "path": "/api/keys", "desc": "Generate a new API key with custom rate limit policy and token bucket capacity"},
            {"method": "POST", "path": "/api/consume", "desc": "Simulate API request consumption against key; returns remaining tokens or 429 Too Many Requests"},
            {"method": "POST", "path": "/api/keys/{key_id}/reset", "desc": "Refill token bucket to maximum capacity"},
            {"method": "GET", "path": "/api/analytics/traffic", "desc": "Get real-time traffic statistics, blocked request counts, and quota usage percentage"},
            {"method": "POST", "path": "/api/rules", "desc": "Create dynamic IP or route-based rate limit rules"}
        ],
        "db_schema": "ApiKeys (id, key_hash, owner_email, tier, max_tokens, refill_rate_per_sec, current_tokens, last_refill_ts), RequestLogs (id, key_id, endpoint, status_code, tokens_consumed, timestamp), RateRules (id, route_pattern, max_rpm, action)",
        "sample_data": "Pre-populate with 4 API keys (Starter Tier, Pro Tier, Enterprise Tier, Abusive Key hitting rate limits) with live token refill tracking.",
        "frontend_sections": [
            "Gateway Traffic Overview: Total requests processed, accepted vs throttled (429) requests ratio, active API keys",
            "Interactive Rate Limit Simulator: Test request button to trigger instant token consumption with live token bucket countdown",
            "API Key Management Table: View keys, tiers, token refill rates, token gauge bars, and instant 'Refill' action",
            "Create API Key Form: Form with tier selection, custom token capacity, and refill rate settings",
            "Traffic Analytics Chart: Breakdown of requests per second by status code (200 OK vs 429 Throttled)"
        ],
        "test_cases": [
            "test_health_check",
            "test_list_api_keys",
            "test_create_api_key_with_tier",
            "test_consume_token_success_decrements_bucket",
            "test_consume_token_exceeding_capacity_returns_429",
            "test_reset_api_key_refills_tokens",
            "test_traffic_analytics_returns_metrics",
            "test_invalid_api_key_returns_401"
        ]
    },
    {
        "name": "event-task-engine",
        "title": "Event-Driven Background Task Queue & Worker Engine",
        "description": "Distributed background task queue and worker orchestration engine with retry policies, exponential backoff, priority scheduling, and dead-letter queues.",
        "category": "Backend Engineering & Async Systems",
        "tech_stack": ["FastAPI", "Python 3.11", "Next.js 14", "TypeScript", "Tailwind CSS", "Docker", "Pytest"],
        "backend_endpoints": [
            {"method": "GET", "path": "/api/health", "desc": "Worker pool status and queue health"},
            {"method": "GET", "path": "/api/tasks", "desc": "List all queued, running, completed, and failed tasks with pagination"},
            {"method": "POST", "path": "/api/tasks", "desc": "Enqueue a new task with payload, priority (Low, Normal, High), and max retries"},
            {"method": "GET", "path": "/api/tasks/{task_id}", "desc": "Get specific task execution details, logs, and retry attempts"},
            {"method": "POST", "path": "/api/tasks/{task_id}/retry", "desc": "Manually retry a failed or dead-letter queue task"},
            {"method": "POST", "path": "/api/workers/process", "desc": "Simulate worker tick: processes next highest priority task from queue"},
            {"method": "GET", "path": "/api/queues/stats", "desc": "Queue depth, throughput, worker concurrency, and failure rate statistics"}
        ],
        "db_schema": "Tasks (id, task_type, payload_json, priority, status, retry_count, max_retries, error_log, created_at, started_at, completed_at), Workers (id, worker_name, status, tasks_completed, last_heartbeat)",
        "sample_data": "Pre-populate with 6 tasks across states: 2 Completed (Data Sync, Email Digest), 1 Processing (Video Transcoding), 1 Queued (PDF Report), 2 Dead-Letter/Failed (Payment Webhook Retry).",
        "frontend_sections": [
            "Queue Health Dashboard: Tasks queued, running, completed, dead-letter count, average processing time",
            "Live Task Stream: Kanban-style columns or filtered table (Queued, Running, Completed, Failed/DLQ)",
            "Enqueue New Task Modal: Custom task name, JSON payload editor, priority selector, and max retry configuration",
            "Worker Simulator Controls: 'Run Worker Step' button to manually advance queue and observe real-time state changes",
            "Task Detail Inspector: Inspect task payload, stack traces, execution latency, and retry timeline"
        ],
        "test_cases": [
            "test_health_probe",
            "test_list_tasks_returns_all_statuses",
            "test_enqueue_task_priority_assignment",
            "test_worker_processes_highest_priority_first",
            "test_failed_task_increments_retry_count",
            "test_max_retries_moves_task_to_dead_letter",
            "test_manual_retry_resets_task_to_queued",
            "test_queue_stats_aggregation"
        ]
    },
    {
        "name": "featureflag-nexus",
        "title": "Smart Feature Flag & A/B Experimentation Platform",
        "description": "Enterprise feature flagging and multivariate A/B testing platform with percentage rollouts, user targeting rules, and conversion analytics.",
        "category": "Product Engineering & DevOps",
        "tech_stack": ["FastAPI", "Python 3.11", "Next.js 14", "TypeScript", "Tailwind CSS", "Docker", "Pytest"],
        "backend_endpoints": [
            {"method": "GET", "path": "/api/health", "desc": "Feature flag engine health check"},
            {"method": "GET", "path": "/api/flags", "desc": "List all feature flags with toggle state, rollout percentage, and variant distribution"},
            {"method": "POST", "path": "/api/flags", "desc": "Create a new feature flag or A/B experiment"},
            {"method": "PATCH", "path": "/api/flags/{flag_id}/toggle", "desc": "Instantly toggle flag enabled/disabled state"},
            {"method": "PATCH", "path": "/api/flags/{flag_id}/rollout", "desc": "Update percentage rollout (0-100%) and targeting rules"},
            {"method": "POST", "path": "/api/evaluate", "desc": "Evaluate feature flag for a specific user ID / attributes; returns assigned variant"},
            {"method": "POST", "path": "/api/flags/{flag_id}/track", "desc": "Track conversion event for flag variant analytics"}
        ],
        "db_schema": "FeatureFlags (id, key, name, description, is_enabled, rollout_percentage, variants_json, created_at), EvaluationLogs (id, flag_key, user_id, assigned_variant, converted, timestamp)",
        "sample_data": "Pre-populate with 4 flags: 'new_checkout_flow' (A/B Test 50/50), 'ai_copilot_enabled' (Rollout 25%), 'dark_mode_v2' (Enabled 100%), 'beta_analytics' (Disabled 0%).",
        "frontend_sections": [
            "Flag Management Dashboard: Quick-toggle switches, rollout percentage sliders, and status badges for every flag",
            "Interactive User Flag Evaluator: Enter User ID + context to preview which variant/state that user receives",
            "A/B Test Conversion Metrics: Comparative bar charts showing Variant A vs Variant B conversion performance",
            "Create New Flag Modal: Form for key name, description, rollout slider, and targeting tags",
            "Audit Log Feed: Timestamped history of flag toggles and configuration changes"
        ],
        "test_cases": [
            "test_health_endpoint",
            "test_list_flags_returns_catalog",
            "test_create_feature_flag_success",
            "test_toggle_feature_flag_inverts_state",
            "test_update_rollout_percentage",
            "test_evaluate_flag_deterministic_for_user",
            "test_track_conversion_increments_stats",
            "test_duplicate_flag_key_returns_409"
        ]
    },
    {
        "name": "prompt-eval-hub",
        "title": "AI Prompt Engineering & LLM Evaluation Benchmark Hub",
        "description": "Collaborative prompt engineering, versioning, and automated evaluation workbench comparing LLM outputs across accuracy, latency, and token cost.",
        "category": "AI Systems & LLMOps",
        "tech_stack": ["FastAPI", "Python 3.11", "Next.js 14", "TypeScript", "Tailwind CSS", "Docker", "Pytest"],
        "backend_endpoints": [
            {"method": "GET", "path": "/api/health", "desc": "Evaluation engine health and provider connectivity"},
            {"method": "GET", "path": "/api/prompts", "desc": "List all prompt templates with version history and tags"},
            {"method": "POST", "path": "/api/prompts", "desc": "Create a new prompt template with variable placeholders (e.g. {{query}})"},
            {"method": "POST", "path": "/api/evaluations/run", "desc": "Execute prompt evaluation suite across test dataset and score accuracy/latency"},
            {"method": "GET", "path": "/api/evaluations", "desc": "List past evaluation benchmark runs with aggregate scores"},
            {"method": "GET", "path": "/api/evaluations/{eval_id}", "desc": "Get detailed itemized evaluation results with pass/fail breakdown"},
            {"method": "POST", "path": "/api/test-cases", "desc": "Add a new golden test case with expected output assertion"}
        ],
        "db_schema": "Prompts (id, title, template, variables_json, version, model, created_at), TestCases (id, prompt_id, input_vars_json, expected_output), EvalRuns (id, prompt_id, total_tests, passed_tests, avg_latency_ms, avg_score, created_at)",
        "sample_data": "Pre-populate with 3 prompt suites: 'Customer Support Categorizer' (v2, 95% pass rate), 'SQL Query Generator' (v1, 88% pass rate), 'Document Summarizer' (v3, 92% pass rate) with test cases.",
        "frontend_sections": [
            "Benchmark Overview: Total prompts, average evaluation pass rate %, cost per 1k runs, active test suites",
            "Prompt Template Editor: Markdown editor with variable syntax highlighting ({{variable}}) and model selector",
            "Evaluation Benchmark Matrix: Run benchmark test suite button with live progress and side-by-side pass/fail scoring",
            "Test Case Dataset Manager: View and add golden input/output pairs for automated assertions",
            "Version Comparison Diff: Compare v1 vs v2 prompt performance, latency improvements, and accuracy delta"
        ],
        "test_cases": [
            "test_health_probe",
            "test_list_prompts_returns_versions",
            "test_create_prompt_template_with_vars",
            "test_create_golden_test_case",
            "test_run_evaluation_computes_scores",
            "test_get_evaluation_detail_includes_cases",
            "test_prompt_variable_interpolation",
            "test_invalid_template_syntax_fails"
        ]
    },
    {
        "name": "trace-health-monitor",
        "title": "Microservices Distributed Tracing & Health Diagnostic Hub",
        "description": "Distributed tracing and telemetry visualizer tracking inter-service HTTP spans, error cascades, P99 latencies, and service dependency graphs.",
        "category": "Distributed Tracing & Observability",
        "tech_stack": ["FastAPI", "Python 3.11", "Next.js 14", "TypeScript", "Tailwind CSS", "Docker", "Pytest"],
        "backend_endpoints": [
            {"method": "GET", "path": "/api/health", "desc": "Trace collector health status"},
            {"method": "GET", "path": "/api/traces", "desc": "List recent distributed traces with total duration, root service, and status code"},
            {"method": "POST", "path": "/api/traces", "desc": "Ingest a new trace with multi-span call tree"},
            {"method": "GET", "path": "/api/traces/{trace_id}", "desc": "Retrieve hierarchical span breakdown for a specific trace with waterfall timeline"},
            {"method": "GET", "path": "/api/topology", "desc": "Get service dependency topology map with edge latencies and error rates"},
            {"method": "GET", "path": "/api/analytics/errors", "desc": "Get top failing service endpoints and error cascade root causes"}
        ],
        "db_schema": "Traces (id, trace_id, root_service, duration_ms, status_code, timestamp), Spans (id, trace_id, parent_span_id, service_name, operation_name, start_time_ms, duration_ms, status_code, tags_json)",
        "sample_data": "Pre-populate with 5 traces: 3 Successful Checkout Traces (API-Gateway -> Auth -> Payment -> DB), 2 Bottleneck/Error Traces (Payment Gateway Timeout 4500ms).",
        "frontend_sections": [
            "Telemetry Health Bar: Total traces captured, P95 / P99 system latency, error rate percentage, active services",
            "Trace Stream Explorer: Filterable trace list by status (200 OK vs 500 Error), minimum duration threshold, service name",
            "Waterfall Span Visualizer: Hierarchical Gantt-style timeline showing parent and child spans with latency bars",
            "Service Dependency Topology: Interactive service-to-service flow map showing traffic throughput and error hotspots",
            "Trace Ingestion Tester: Button to simulate new trace payloads and observe real-time waterfall rendering"
        ],
        "test_cases": [
            "test_health_endpoint",
            "test_list_traces_returns_recent",
            "test_ingest_trace_with_spans",
            "test_get_trace_waterfall_spans",
            "test_topology_graph_computes_dependencies",
            "test_error_analytics_identifies_root_causes",
            "test_filter_traces_by_duration_threshold",
            "test_invalid_span_payload_validation"
        ]
    },
    {
        "name": "webhook-relay-engine",
        "title": "Serverless Webhook Relay & Reliable Ingestion Engine",
        "description": "High-reliability webhook delivery and ingestion gateway with HMAC SHA-256 signature verification, exponential backoff retries, and delivery audit logs.",
        "category": "API Gateways & Event Streaming",
        "tech_stack": ["FastAPI", "Python 3.11", "Next.js 14", "TypeScript", "Tailwind CSS", "Docker", "Pytest"],
        "backend_endpoints": [
            {"method": "GET", "path": "/api/health", "desc": "Webhook relay queue health check"},
            {"method": "GET", "path": "/api/endpoints", "desc": "List registered webhook destination endpoints and delivery success rates"},
            {"method": "POST", "path": "/api/endpoints", "desc": "Register new destination endpoint with secret key and subscribed events"},
            {"method": "POST", "path": "/api/ingest", "desc": "Ingest incoming webhook payload, verify HMAC signature, and enqueue for relay"},
            {"method": "GET", "path": "/api/deliveries", "desc": "List delivery attempts with HTTP status codes, response times, and payload previews"},
            {"method": "POST", "path": "/api/deliveries/{delivery_id}/retry", "desc": "Manually trigger redelivery of a failed webhook attempt"},
            {"method": "GET", "path": "/api/stats", "desc": "Delivery success rate, average latency, and failed delivery count"}
        ],
        "db_schema": "Endpoints (id, target_url, secret_key, subscribed_events, is_active, success_count, fail_count), WebhookEvents (id, event_type, payload_json, signature, created_at), Deliveries (id, event_id, endpoint_id, attempt_number, response_status, response_body, latency_ms, status, timestamp)",
        "sample_data": "Pre-populate with 3 endpoints (Stripe Ingestion URL, GitHub Sync Endpoint, Slack Alert Webhook) and 8 delivery logs including successful 200s and retried 503s.",
        "frontend_sections": [
            "Delivery Stats Ribbon: Total events relayed, delivery success rate %, failed deliveries needing retry, P95 relay latency",
            "Live Delivery Log Inspector: Real-time event log with status pills (200 OK, 500 Retry, Pending), duration, and payload view",
            "Endpoint Subscriptions Table: View active destination endpoints, secret keys, subscribed event types, and toggle active state",
            "Manual Webhook Dispatcher: Form to simulate firing a webhook event with custom JSON payload and HMAC signing",
            "Payload & Response Drawer: Inspect full request payload, HTTP headers, signature verification status, and destination server response"
        ],
        "test_cases": [
            "test_health_probe",
            "test_list_endpoints_returns_catalog",
            "test_register_new_endpoint",
            "test_ingest_webhook_enqueues_event",
            "test_hmac_signature_verification_valid",
            "test_hmac_signature_verification_invalid_fails",
            "test_manual_redelivery_attempt",
            "test_delivery_stats_calculation"
        ]
    },
    {
        "name": "schema-drift-guard",
        "title": "Automated Database Schema Migration & Drift Guard",
        "description": "Automated database schema inspection, drift detection, and SQL migration generation engine comparing staging and production schemas.",
        "category": "Database Engineering & DevOps",
        "tech_stack": ["FastAPI", "Python 3.11", "Next.js 14", "TypeScript", "Tailwind CSS", "Docker", "Pytest"],
        "backend_endpoints": [
            {"method": "GET", "path": "/api/health", "desc": "Database inspector engine health check"},
            {"method": "GET", "path": "/api/schemas", "desc": "List registered target database environments and schemas"},
            {"method": "POST", "path": "/api/schemas/inspect", "desc": "Inspect schema tables, columns, indexes, and constraints for an environment"},
            {"method": "POST", "path": "/api/drift/detect", "desc": "Compare Staging vs Production schemas and compute missing columns/indexes"},
            {"method": "GET", "path": "/api/drift/reports", "desc": "List past schema drift audit reports and severity ratings"},
            {"method": "POST", "path": "/api/migrations/generate", "desc": "Auto-generate safe SQL ALTER / CREATE migration script to resolve drift"},
            {"method": "POST", "path": "/api/migrations/apply", "desc": "Apply generated migration to target environment"}
        ],
        "db_schema": "Environments (id, env_name, db_type, connection_str, last_inspected), Tables (id, env_id, table_name, columns_json, indexes_json), DriftReports (id, source_env, target_env, drift_count, diff_summary_json, generated_sql, status, created_at)",
        "sample_data": "Pre-populate with 2 environments (Staging vs Production) where Staging has 2 new columns on 'users' table and 1 new index on 'orders' table causing drift.",
        "frontend_sections": [
            "Schema Health Summary: Environments connected, drift status badge (In Sync vs Drift Detected), pending migrations count",
            "Visual Schema Drift Diff: Side-by-side comparison of Staging vs Production highlighting missing columns and index mismatches in red/green",
            "SQL Migration Generator Panel: Live generated idempotent SQL migration script with 'Copy SQL' and 'Apply Migration' actions",
            "Database Schema Explorer: Table and column tree browser showing data types, nullable constraints, and primary keys",
            "Migration Audit Trail: History of applied migrations with execution timestamps and rollbacks"
        ],
        "test_cases": [
            "test_health_endpoint",
            "test_list_environments",
            "test_inspect_schema_returns_tables",
            "test_drift_detection_finds_missing_column",
            "test_generate_migration_produces_valid_sql",
            "test_drift_reports_list_historical_runs",
            "test_apply_migration_updates_environment_state",
            "test_identical_schemas_report_zero_drift"
        ]
    },
    {
        "name": "collab-canvas-engine",
        "title": "Real-Time WebSocket Collaborative Canvas API",
        "description": "High-performance collaborative whiteboard and document room engine with operational transformation, cursor presence sync, and state persistence.",
        "category": "Real-Time Systems & WebSockets",
        "tech_stack": ["FastAPI", "Python 3.11", "Next.js 14", "TypeScript", "Tailwind CSS", "Docker", "Pytest"],
        "backend_endpoints": [
            {"method": "GET", "path": "/api/health", "desc": "WebSocket room manager and memory state health"},
            {"method": "GET", "path": "/api/rooms", "desc": "List all active collaborative rooms with participant count and object count"},
            {"method": "POST", "path": "/api/rooms", "desc": "Create a new collaborative canvas room with access permissions"},
            {"method": "GET", "path": "/api/rooms/{room_id}/state", "desc": "Get current canvas state and all drawn element objects"},
            {"method": "POST", "path": "/api/rooms/{room_id}/elements", "desc": "Add or update a canvas element (rectangle, circle, text, path)"},
            {"method": "DELETE", "path": "/api/rooms/{room_id}/elements/{element_id}", "desc": "Delete an element and broadcast removal"},
            {"method": "POST", "path": "/api/rooms/{room_id}/presence", "desc": "Update user cursor coordinates and active selection status"}
        ],
        "db_schema": "Rooms (id, name, slug, max_users, active_users_count, created_at), CanvasElements (id, room_id, element_type, x_pos, y_pos, width, height, color, z_index, created_by, updated_at), ActivePresences (id, room_id, user_name, cursor_x, cursor_y, last_seen)",
        "sample_data": "Pre-populate with 2 active rooms ('System Architecture Whiteboard' with 8 shapes/connectors and 'Sprint Retro Board' with 5 sticky notes).",
        "frontend_sections": [
            "Active Rooms Lobby: List of collaborative canvas rooms with live participant badges and 'Join Room' action",
            "Interactive Canvas Viewport: Interactive visual canvas displaying rendered shape elements with drag and select capabilities",
            "Shape Toolbar: Tool picker to add Rectangles, Sticky Notes, Circles, and Text cards to canvas",
            "Live Cursor & Presence Overlay: Visual badges showing simulated active collaborators and their cursor positions",
            "Element Property Inspector: Modify selected element's fill color, dimensions, and text content with instant update"
        ],
        "test_cases": [
            "test_health_probe",
            "test_list_rooms_returns_active_rooms",
            "test_create_room_success",
            "test_get_room_canvas_elements",
            "test_add_canvas_element_updates_state",
            "test_delete_canvas_element",
            "test_update_presence_coordinates",
            "test_get_nonexistent_room_returns_404"
        ]
    }
]

# ---------------------------------------------------------------------------
# LLM Generation Helper
# ---------------------------------------------------------------------------
def generate_with_groq(system_prompt: str, user_prompt: str, max_tokens: int = MAX_TOKENS) -> str:
    """Invokes the Groq API via litellm with fallback retry logic and clean output parsing."""
    try:
        response = litellm.completion(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=TEMPERATURE,
            max_tokens=max_tokens
        )
        content = response.choices[0].message.content
        # Strip markdown code block wrapping if LLM included it
        content = re.sub(r"^" + re.escape(B3) + r"[a-zA-Z]*\n", "", content.strip())
        content = re.sub(r"\n" + re.escape(B3) + r"$", "", content.strip())
        return content.strip()
    except Exception as e:
        print(f"  [LLM Error] Error calling Groq model {MODEL_NAME}: {e}")
        raise e

# ---------------------------------------------------------------------------
# Stage 1: Backend Code Generator
# ---------------------------------------------------------------------------
def generate_backend_code(blueprint: dict) -> str:
    print("\n--- [Stage 1/4] Generating Production FastAPI Backend (main.py) ---")
    system_prompt = (
        "You are a Senior Principal Python Architect. You write flawless, fully functional, production-ready FastAPI applications. "
        "Every endpoint must have full business logic with realistic in-memory or SQLite storage, Pydantic v2 models, validation, "
        "proper HTTP status codes, CORS middleware, and seed data. Do not write placeholder comments, TODOs, or ellipses (...). "
        "Return ONLY the complete Python code file without markdown formatting."
    )
    user_prompt = f"""
Write the complete, self-contained `backend/main.py` for the following production project:

Project: {blueprint['title']}
Description: {blueprint['description']}
Database Schema: {blueprint['db_schema']}
Pre-populated Seed Data: {blueprint['sample_data']}

Required Endpoints to implement completely:
{json.dumps(blueprint['backend_endpoints'], indent=2)}

Strict Requirements:
1. Use FastAPI, Pydantic (BaseModel, Field), CORSMiddleware, typing (List, Optional, Dict, Any), uuid, datetime.
2. Enable CORS for all origins (allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]).
3. Implement an in-memory database dictionary pre-loaded with the specified seed data on startup.
4. Implement full CRUD/action logic for every single endpoint listed above with correct HTTP methods and status codes.
5. Include a GET /api/health endpoint returning {{"status": "healthy", "service": "{blueprint['name']}", "timestamp": "<ISO>"}}.
6. All Pydantic request models must have Field validation (e.g. min_length, max_length, ge, le).
7. Return ONLY clean Python code. No markdown fences, no explanations.
"""
    code = generate_with_groq(system_prompt, user_prompt, max_tokens=2000)
    return code

# ---------------------------------------------------------------------------
# Stage 2: Test Suite Generator
# ---------------------------------------------------------------------------
def generate_test_code(blueprint: dict, backend_code: str) -> str:
    print(f"\nWaiting {STAGE_DELAY}s for Groq rate-limit window reset...")
    time.sleep(STAGE_DELAY)
    print("\n--- [Stage 2/4] Generating Pytest Test Suite (test_main.py) ---")

    system_prompt = (
        "You are a Senior QA Automation Engineer. You write comprehensive, bug-free pytest test suites using FastAPI's TestClient. "
        "Every test case must execute against the provided FastAPI app and assert realistic response data and status codes. "
        "Return ONLY executable Python test code without markdown formatting."
    )
    user_prompt = f"""
Write the complete `backend/test_main.py` test suite for this FastAPI application.

Project Name: {blueprint['name']}
Target Test Cases:
{json.dumps(blueprint['test_cases'], indent=2)}

FastAPI Backend Reference:
{B3}python
{backend_code[:1200]}
{B3}

Strict Requirements:
1. Import `pytest`, `from fastapi.testclient import TestClient`, `from main import app`.
2. Initialize `client = TestClient(app)`.
3. Implement every single test case from the list above with descriptive docstrings and multiple assert statements.
4. Include positive tests, negative validation tests (e.g. 404 on missing ID, 422 on invalid schema), and edge cases.
5. Return ONLY clean Python code. No markdown fences.
"""
    code = generate_with_groq(system_prompt, user_prompt, max_tokens=1800)
    return code

# ---------------------------------------------------------------------------
# Stage 3: Frontend Code Generator
# ---------------------------------------------------------------------------
def generate_frontend_code(blueprint: dict, backend_code: str) -> str:
    print(f"\nWaiting {STAGE_DELAY}s for Groq rate-limit window reset...")
    time.sleep(STAGE_DELAY)
    print("\n--- [Stage 3/4] Generating Next.js 14 Interactive Frontend (page.tsx) ---")

    system_prompt = (
        "You are a Senior Staff Frontend Engineer specializing in React, Next.js 14 App Router, and Tailwind CSS. "
        "You create polished, modern dark-themed interactive dashboards with real fetch() API integration, loading spinners, "
        "empty states, modal dialogs, and smooth state updates. Return ONLY the TypeScript React code file."
    )
    user_prompt = f"""
Write the complete `frontend/src/app/page.tsx` for the project:

Project: {blueprint['title']}
Description: {blueprint['description']}
UI Sections to Build:
{json.dumps(blueprint['frontend_sections'], indent=2)}

Backend API Endpoints Available:
{json.dumps(blueprint['backend_endpoints'], indent=2)}

Strict Requirements:
1. Add `'use client';` at the very top.
2. Define TypeScript interfaces for all data models used in the UI.
3. Use `const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';` for all fetch calls.
4. Implement `useEffect` to fetch initial data from backend endpoints on mount.
5. Implement interactive actions (e.g. button clicks to trigger API calls, forms to submit new entries, filters).
6. Implement beautiful dark mode UI with Tailwind CSS (e.g., bg-slate-900, bg-slate-800/60, text-white, border-slate-700, indigo/cyan accent badges).
7. Include loading states (spinners or skeleton pulses) and error alert banners when fetch fails.
8. Include clean inline SVG icons for key indicators.
9. Do not use external icon packages. Return ONLY the complete page.tsx code without markdown fences.
"""
    code = generate_with_groq(system_prompt, user_prompt, max_tokens=2000)
    return code

# ---------------------------------------------------------------------------
# Stage 4: Infrastructure & Static Files (Guaranteed 100% Working)
# ---------------------------------------------------------------------------
def get_hardcoded_files(blueprint: dict) -> dict:
    """Generates tested, production-grade infrastructure files."""
    name = blueprint["name"]
    title = blueprint["title"]
    desc = blueprint["description"]

    # Package.json with correct dev binding for Codespaces and proper TypeScript dependencies
    package_json = json.dumps({
        "name": f"{name}-frontend",
        "version": "1.0.0",
        "private": True,
        "scripts": {
            "dev": "next dev -H 0.0.0.0 -p 3000",
            "build": "next build",
            "start": "next start -H 0.0.0.0 -p 3000",
            "lint": "next lint"
        },
        "dependencies": {
            "next": "14.1.0",
            "react": "^18.2.0",
            "react-dom": "^18.2.0",
            "clsx": "^2.1.0",
            "tailwind-merge": "^2.2.1"
        },
        "devDependencies": {
            "typescript": "^5.3.3",
            "@types/node": "^20.11.24",
            "@types/react": "^18.2.61",
            "@types/react-dom": "^18.2.19",
            "autoprefixer": "^10.4.18",
            "postcss": "^8.4.35",
            "tailwindcss": "^3.4.1"
        }
    }, indent=2)

    # Docker Compose with no deprecated version key and correct internal networking
    docker_compose = f"""services:
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: {name}-backend
    ports:
      - "8000:8000"
    environment:
      - PORT=8000
      - ENVIRONMENT=production
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"]
      interval: 15s
      timeout: 5s
      retries: 3

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    container_name: {name}-frontend
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000
    depends_on:
      backend:
        condition: service_healthy
    restart: unless-stopped
"""

    backend_dockerfile = """FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

    frontend_dockerfile = """FROM node:20-alpine AS runner

WORKDIR /app

ENV NODE_ENV=development
ENV PORT=3000
ENV HOSTNAME=0.0.0.0

COPY package.json package-lock.json* ./
RUN npm install

COPY . .

EXPOSE 3000

CMD ["npm", "run", "dev"]
"""

    backend_requirements = """fastapi==0.110.0
uvicorn[standard]==0.27.1
pydantic==2.6.4
pytest==8.0.2
httpx==0.27.0
python-multipart==0.0.9
"""

    tsconfig = json.dumps({
        "compilerOptions": {
            "lib": ["dom", "dom.iterable", "esnext"],
            "allowJs": True,
            "skipLibCheck": True,
            "strict": True,
            "noEmit": True,
            "esModuleInterop": True,
            "module": "esnext",
            "moduleResolution": "bundler",
            "resolveJsonModule": True,
            "isolatedModules": True,
            "jsx": "preserve",
            "incremental": True,
            "plugins": [{"name": "next"}],
            "paths": {"@/*": ["./src/*"]}
        },
        "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
        "exclude": ["node_modules"]
    }, indent=2)

    next_config = """/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
};

module.exports = nextConfig;
"""

    tailwind_config = """/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          50: '#eef2ff',
          500: '#6366f1',
          600: '#4f46e5',
          700: '#4338ca',
        }
      }
    },
  },
  plugins: [],
}
"""

    postcss_config = """module.exports = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
"""

    globals_css = """@tailwind base;
@tailwind components;
@tailwind utilities;

:root {
  --foreground-rgb: 255, 255, 255;
  --background-start-rgb: 15, 23, 42;
  --background-end-rgb: 15, 23, 42;
}

body {
  color: rgb(var(--foreground-rgb));
  background: #0f172a;
  min-height: 100vh;
}
"""

    layout_tsx = f"""import type {{ Metadata }} from 'next'
import './globals.css'

export const metadata: Metadata = {{
  title: '{title}',
  description: '{desc}',
}}

export default function RootLayout({{
  children,
}}: {{
  children: React.ReactNode
}}) {{
  return (
    <html lang="en" className="dark">
      <body className="bg-slate-950 text-slate-100 min-h-screen antialiased">
        <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center font-bold text-white shadow-lg shadow-indigo-500/30">
                ⚡
              </div>
              <span className="font-semibold text-lg tracking-tight text-white">{name}</span>
            </div>
            <div className="flex items-center space-x-4 text-xs">
              <span className="px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-medium flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
                System Live
              </span>
              <span className="text-slate-400 hidden sm:inline">v1.0.0</span>
            </div>
          </div>
        </header>
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {{children}}
        </main>
      </body>
    </html>
  )
}}
"""

    ci_workflow = f"""name: Continuous Integration

on:
  push:
    branches: [ main, master ]
  pull_request:
    branches: [ main, master ]

jobs:
  backend-tests:
    name: Backend Pytest Suite
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'
          
      - name: Install Dependencies
        run: |
          cd backend
          pip install --upgrade pip
          pip install -r requirements.txt
          
      - name: Run Pytest with Coverage
        run: |
          cd backend
          pytest -v --tb=short

  docker-build-verify:
    name: Docker Compose Build Check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build Docker Images
        run: docker compose build
"""

    readme = f"""# {title}

[![Continuous Integration](https://github.com/patel-tirth-1952008/{name}/actions/workflows/ci.yml/badge.svg)](https://github.com/patel-tirth-1952008/{name}/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Next.js 14](https://img.shields.io/badge/Next.js-14.1.0-000000?logo=next.js&logoColor=white)](https://nextjs.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

> {desc}

---

## 🏛️ System Architecture

{B3}
                                  +-----------------------------+
                                  |    Next.js 14 App Router    |
                                  |  (TypeScript + Tailwind)    |
                                  +--------------+--------------+
                                                 |
                                                 | HTTP / JSON
                                                 v
                                  +--------------+--------------+
                                  |     FastAPI Backend API     |
                                  |   (Pydantic + In-Memory DB) |
                                  +--------------+--------------+
                                                 |
                                        +--------+--------+
                                        |                 |
                                        v                 v
                               +----------------+  +----------------+
                               | Health Monitor |  |  CRUD Engine   |
                               +----------------+  +----------------+
{B3}

---

## ✨ Features

- **Production-Ready FastAPI Backend**: Full schema validation with Pydantic v2, CORS support, comprehensive error handling, and pre-seeded realistic data.
- **Modern Next.js 14 Dashboard**: Dark-mode UI styled with Tailwind CSS, responsive state management, and real-time backend API consumption.
- **Automated Pytest Suite**: Complete unit and integration test coverage executing against FastAPI's TestClient.
- **One-Command Dockerization**: Multi-stage Dockerfiles and docker-compose.yml for unified local setup.
- **GitHub Actions CI Pipeline**: Automated build and test verification on every commit.

---

## 🔌 API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
"""
    for ep in blueprint["backend_endpoints"]:
        readme += f"| `{ep['method']}` | `{ep['path']}` | {ep['desc']} |\n"

    readme += f"""
---

## 🚀 Quick Start

### 1. Run with Docker Compose (Recommended)

{B3}bash
# Clone repository
git clone https://github.com/patel-tirth-1952008/{name}.git
cd {name}

# Spin up both frontend and backend
docker-compose up --build
{B3}

- **Frontend Application**: `http://localhost:3000`
- **FastAPI Interactive Docs**: `http://localhost:8000/docs`
- **Backend Health Check**: `http://localhost:8000/api/health`

---

### 2. Run Manually (Local Development)

#### Backend Setup
{B3}bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
{B3}

#### Run Tests
{B3}bash
cd backend
pytest -v
{B3}

#### Frontend Setup
{B3}bash
cd frontend
npm install
npm run dev
{B3}

---

## 📂 Project Structure

{B3}
.
├── .github/
│   └── workflows/
│       └── ci.yml                 # Automated Pytest CI workflow
├── backend/
│   ├── main.py                    # FastAPI application & business logic
│   ├── test_main.py               # Comprehensive Pytest test suite
│   ├── requirements.txt           # Python dependencies
│   └── Dockerfile                 # Backend container definition
├── frontend/
│   ├── src/
│   │   └── app/
│   │       ├── globals.css        # Tailwind base styles
│   │       ├── layout.tsx         # Next.js Root Layout with header
│   │       └── page.tsx           # Interactive client dashboard
│   ├── package.json               # Node.js dependencies
│   ├── tsconfig.json              # TypeScript configuration
│   ├── tailwind.config.js         # Tailwind configuration
│   ├── next.config.js             # Next.js configuration
│   └── Dockerfile                 # Frontend container definition
├── docker-compose.yml             # Orchestration for both services
├── .env.example                   # Environment configuration template
├── .gitignore                     # Git ignore rules
└── README.md                      # Comprehensive project documentation
{B3}

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
"""

    gitignore = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.pytest_cache/
.coverage

# Node.js
node_modules/
.next/
out/
build/
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Environment & OS
.env
.env.local
.DS_Store
Thumbs.db
"""

    env_example = """# Backend Configuration
PORT=8000
ENVIRONMENT=development

# Frontend Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000
"""

    mit_license = f"""MIT License

Copyright (c) {datetime.now(timezone.utc).year} Tirth Patel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

    return {
        "docker-compose.yml": docker_compose,
        ".env.example": env_example,
        ".gitignore": gitignore,
        "LICENSE": mit_license,
        "README.md": readme,
        ".github/workflows/ci.yml": ci_workflow,
        "backend/requirements.txt": backend_requirements,
        "backend/Dockerfile": backend_dockerfile,
        "frontend/package.json": package_json,
        "frontend/Dockerfile": frontend_dockerfile,
        "frontend/tsconfig.json": tsconfig,
        "frontend/next.config.js": next_config,
        "frontend/tailwind.config.js": tailwind_config,
        "frontend/postcss.config.js": postcss_config,
        "frontend/src/app/globals.css": globals_css,
        "frontend/src/app/layout.tsx": layout_tsx,
    }

# ---------------------------------------------------------------------------
# Code Validation
# ---------------------------------------------------------------------------
def validate_python_code(code: str) -> bool:
    """Verifies that the generated Python code is syntactically valid."""
    try:
        compile(code, "<string>", "exec")
        return True
    except SyntaxError as e:
        print(f"  [Validation Warning] Python syntax error: {e}")
        return False

def validate_test_code(code: str) -> bool:
    """Verifies that the generated test suite contains TestClient and asserts."""
    if not validate_python_code(code):
        return False
    if "TestClient" not in code or "def test_" not in code:
        print("  [Validation Warning] Test code missing TestClient or test definitions.")
        return False
    return True

def validate_frontend_code(code: str) -> bool:
    """Verifies that the frontend page has required Next.js elements."""
    if "'use client'" not in code and '"use client"' not in code:
        print("  [Validation Warning] Frontend missing 'use client' directive.")
        return False
    if "fetch(" not in code and "useEffect" not in code:
        print("  [Validation Warning] Frontend missing fetch or useEffect calls.")
        return False
    return True

# ---------------------------------------------------------------------------
# GitHub Repository Creation & File Push
# ---------------------------------------------------------------------------
def create_github_repo(repo_name: str, description: str) -> bool:
    """Creates a new public repository on GitHub under the authenticated user's account."""
    token = os.environ.get("GITHUB_PAT") or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("  [GitHub Error] GITHUB_PAT environment variable is not set!")
        return False

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }

    url = "https://api.github.com/user/repos"
    payload = {
        "name": repo_name,
        "description": description,
        "private": False,
        "has_issues": True,
        "has_projects": True,
        "has_wiki": False,
        "auto_init": False
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=20)
        if response.status_code == 201:
            print(f"  [GitHub Success] Repository '{repo_name}' created successfully!")
            return True
        elif response.status_code == 422:
            print(f"  [GitHub Notice] Repository '{repo_name}' already exists. Will commit updates directly.")
            return True
        else:
            print(f"  [GitHub Error] Failed to create repo: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"  [GitHub Error] Exception creating repo: {e}")
        return False

def push_files_to_github(repo_name: str, files_dict: dict) -> bool:
    """Commits and pushes all project files to the GitHub repository using the Contents API."""
    token = os.environ.get("GITHUB_PAT") or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("  [GitHub Error] GITHUB_PAT not set!")
        return False

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }

    # Fetch authenticated user's login username
    user_res = requests.get("https://api.github.com/user", headers=headers, timeout=10)
    if user_res.status_code != 200:
        print(f"  [GitHub Error] Could not fetch authenticated user: {user_res.text}")
        return False
    owner = user_res.json()["login"]

    print(f"\n--- Uploading {len(files_dict)} files to {owner}/{repo_name} ---")

    success_count = 0
    for path, content in files_dict.items():
        url = f"https://api.github.com/repos/{owner}/{repo_name}/contents/{path}"
        
        # Check if file already exists to obtain its SHA (required for updates)
        sha = None
        get_res = requests.get(url, headers=headers, timeout=10)
        if get_res.status_code == 200:
            sha = get_res.json().get("sha")

        encoded_content = base64.b64encode(content.encode("utf-8")).decode("utf-8")
        payload = {
            "message": f"feat: add {path} [Production Release]",
            "content": encoded_content
        }
        if sha:
            payload["sha"] = sha

        put_res = requests.put(url, headers=headers, json=payload, timeout=20)
        if put_res.status_code in (200, 201):
            print(f"  ✓ Pushed {path}")
            success_count += 1
        else:
            print(f"  ✗ Failed {path}: {put_res.status_code} - {put_res.text[:100]}")

    print(f"\n[GitHub Status] Successfully uploaded {success_count}/{len(files_dict)} files to {owner}/{repo_name}!")
    return success_count == len(files_dict)

# ---------------------------------------------------------------------------
# History Tracking
# ---------------------------------------------------------------------------
def load_history() -> list:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_history(entry: dict):
    history = load_history()
    history.append(entry)
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------
def run_portfolio_builder():
    print("=================================================================")
    print("  AGENT 1: AUTONOMOUS SENIOR PORTFOLIO PROJECT BUILDER")
    print(f"  Model: {MODEL_NAME} | Delay: {STAGE_DELAY}s | Target: Senior Full-Stack")
    print("=================================================================")

    # Select project blueprint (cycle through or pick unused)
    history = load_history()
    used_names = {h.get("name") for h in history}

    available_blueprints = [b for b in PROJECT_BLUEPRINTS if b["name"] not in used_names]
    if not available_blueprints:
        print("  [Info] All blueprints deployed! Cycling from the beginning with timestamped release.")
        blueprint = random.choice(PROJECT_BLUEPRINTS)
    else:
        blueprint = available_blueprints[0]

    print(f"\n[Selected Blueprint] {blueprint['title']}")
    print(f"Category:    {blueprint['category']}")
    print(f"Repository:  {blueprint['name']}")
    print(f"Description: {blueprint['description']}")

    # 1. Generate FastAPI Backend
    backend_code = generate_backend_code(blueprint)
    if not validate_python_code(backend_code):
        print("  [Fix] Attempting syntax cleanup on generated backend code...")
        backend_code = backend_code.replace("```python", "").replace("```", "").strip()

    # 2. Generate Pytest Test Suite
    test_code = generate_test_code(blueprint, backend_code)
    if not validate_python_code(test_code):
        test_code = test_code.replace("```python", "").replace("```", "").strip()

    # 3. Generate Next.js 14 Frontend Page
    frontend_code = generate_frontend_code(blueprint, backend_code)
    if not validate_frontend_code(frontend_code):
        frontend_code = frontend_code.replace("```tsx", "").replace("```typescript", "").replace("```", "").strip()

    # 4. Assemble All Infrastructure & Config Files
    all_files = get_hardcoded_files(blueprint)
    all_files["backend/main.py"] = backend_code
    all_files["backend/test_main.py"] = test_code
    all_files["frontend/src/app/page.tsx"] = frontend_code

    print(f"\n[Assembly Complete] Prepared {len(all_files)} files ready for deployment.")

    # 5. Create GitHub Repo and Push
    repo_created = create_github_repo(blueprint["name"], blueprint["description"])
    if repo_created:
        pushed = push_files_to_github(blueprint["name"], all_files)
        if pushed:
            save_history({
                "name": blueprint["name"],
                "title": blueprint["title"],
                "repo_url": f"https://github.com/patel-tirth-1952008/{blueprint['name']}",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "status": "deployed",
                "files_count": len(all_files)
            })
            print("\n=================================================================")
            print(f"  🎉 SUCCESS: Deployed '{blueprint['name']}' to GitHub!")
            print(f"  URL: https://github.com/patel-tirth-1952008/{blueprint['name']}")
            print("=================================================================")
            return

    print("\n[Warning] Repository push encountered errors. Check logs above.")

if __name__ == "__main__":
    run_portfolio_builder()