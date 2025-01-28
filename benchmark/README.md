# Benchmark

Performance benchmarking for Ollama.

## Prerequisites
- Ollama server running locally (`127.0.0.1:11434`)
- Desired models pre-downloaded (e.g., `llama3.2:1b`)

## Run Benchmark
```bash
# Run all tests
go test -bench=. -timeout 30m ./...
```
