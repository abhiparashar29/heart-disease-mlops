#!/usr/bin/env bash
# Sends realistic random patient requests to the API (through the Ingress) so the
# Grafana dashboard has data. Every 10th request is deliberately invalid (-> 422).
# Usage: bash monitoring/generate_traffic.sh [number_of_requests]
set -euo pipefail

N=${1:-200}
IP=$(minikube ip)
URL="http://heart-api.local/predict"
THALS=(3 6 7)
ok=0
bad=0

for i in $(seq 1 "$N"); do
  if (( i % 10 == 0 )); then
    body='{"age": -5, "sex": 1}'
  else
    body=$(printf '{"age":%d,"sex":%d,"cp":%d,"trestbps":%d,"chol":%d,"fbs":%d,"restecg":%d,"thalach":%d,"exang":%d,"oldpeak":%d.%d,"slope":%d,"ca":%d,"thal":%d}' \
      $((30 + RANDOM % 45)) $((RANDOM % 2)) $((1 + RANDOM % 4)) $((100 + RANDOM % 70)) \
      $((160 + RANDOM % 160)) $((RANDOM % 2)) $((RANDOM % 3)) $((90 + RANDOM % 110)) \
      $((RANDOM % 2)) $((RANDOM % 4)) $((RANDOM % 10)) $((1 + RANDOM % 3)) \
      $((RANDOM % 4)) "${THALS[$((RANDOM % 3))]}")
  fi
  code=$(curl -s -o /dev/null -w "%{http_code}" --resolve "heart-api.local:80:${IP}" \
    -X POST "$URL" -H "Content-Type: application/json" -d "$body")
  if [[ "$code" == "200" ]]; then ok=$((ok + 1)); else bad=$((bad + 1)); fi
  printf "\rSent %d/%d  (200 OK: %d, rejected: %d)" "$i" "$N" "$ok" "$bad"
  sleep 0.2
done
echo