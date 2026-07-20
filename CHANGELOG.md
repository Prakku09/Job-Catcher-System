# Changelog

## [1.1.0] - 2026-07-20
### Added
- Task 20 Completion requirements.
- `POST /batch_predict` endpoint to process multiple job matching requests concurrently.
- `GET /metrics` endpoint for exposing basic API performance and usage metrics.
- Global request-level logging middleware to record unique `request_id`, `timestamp`, `latency`, `model_version`, `prediction`, and `errors`.
- Graceful error handling for missing models, empty batch payloads, and internal prediction failures.
- Performance testing and API validation scripts.

### Changed
- `src/app.py` architecture expanded to support new endpoints while fully preserving previous machine learning components and endpoint behavior.
